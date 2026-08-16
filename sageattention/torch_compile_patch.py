"""Small compatibility patches for the local Torch/Inductor build.

The custom-op autotuner currently requires every candidate to produce one
tensor.  Mutation-only custom ops are valid in Torch and are handled by
functionalization, but they have no result layout for the autotuner to use.
This module bridges that representation gap without changing Inductor's
dependency or mutation analysis.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
from torch._inductor.codegen.subgraph import SubgraphTemplate, inline_subgraph_to_ir_nodes
from torch._inductor.kernel import custom_op as custom_op_kernel
from torch._inductor.lowering import validate_ir
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.virtualized import V
from torch.utils import _pytree

_PATCH_ATTRIBUTE = "_sageattention_mutation_only_autotune_patch"
_FUNCTIONALIZE_OPS: set[str] = set()
_FUNCTIONALIZE_LIBRARIES: list[torch.library.Library] = []


def _is_mutation_only_op(op_overload: Any) -> bool:
    schema = op_overload._schema
    returns = schema.returns
    no_return = not returns or (len(returns) == 1 and isinstance(returns[0].type, torch.NoneType))
    has_mutation = any(
        argument.alias_info is not None and argument.alias_info.is_write for argument in schema.arguments
    )
    return no_return and has_mutation


def _first_tensor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError("Mutation-only custom-op autotuning requires a tensor argument.")


def _with_autotune_token(
    decomposition: Callable[..., Any],
) -> Callable[..., torch.Tensor]:
    """Give a mutation-only decomposition a private, dead tensor result."""

    @wraps(decomposition)
    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = decomposition(*args, **kwargs)
        if result is not None:
            decomposition_name = getattr(decomposition, "__name__", type(decomposition).__name__)
            raise RuntimeError(f"Mutation-only custom-op decomposition {decomposition_name!r} must return None.")
        tensor = _first_tensor(args, kwargs)
        return torch.empty((0,), device=tensor.device, dtype=torch.uint8)

    return wrapped


def register_mutation_only_custom_op_functionalization(op_def: Any) -> None:
    """Register a redispatch rule for direct ``torch.func.functionalize`` calls."""

    op_overload = op_def._opoverload
    qualified_name = op_overload._schema.name
    if qualified_name in _FUNCTIONALIZE_OPS:
        return

    namespace, op_name = qualified_name.split("::", 1)
    library = torch.library.Library(namespace, "IMPL", "Functionalize")

    def unwrap(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and torch._is_functional_tensor(value):
            return torch._from_functional_tensor(value)
        return value

    def functionalize_impl(*args: Any, **kwargs: Any) -> Any:
        args_unwrapped = _pytree.tree_map(unwrap, args)
        kwargs_unwrapped = _pytree.tree_map(unwrap, kwargs)
        with torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)):
            return op_overload(*args_unwrapped, **kwargs_unwrapped)

    library.impl(op_name, functionalize_impl)
    _FUNCTIONALIZE_OPS.add(qualified_name)
    _FUNCTIONALIZE_LIBRARIES.append(library)


def _autotune_mutation_only_custom_op(
    *,
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    non_tensor_args: list[dict[str, Any]],
    op_overload: Any,
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None,
    config_patches_list: list[dict[str, Any]] | None,
    min_speedup_threshold: float,
    benchmark_with_cudagraphs: bool,
) -> tuple[Any, Any]:
    if not decompositions:
        raise RuntimeError(f"Mutation-only custom op {name!r} has no autotune choices.")
    if not inputs:
        raise RuntimeError(f"Mutation-only custom op {name!r} requires tensor inputs.")
    if len(decompositions) != len(non_tensor_args):
        raise ValueError(
            "decompositions and non_tensor_args must have the same length, "
            f"got {len(decompositions)} and {len(non_tensor_args)}"
        )

    if config_patches_list is None:
        config_patches_list = [{} for _ in decompositions]

    input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] = {}
    if user_input_gen_fns:
        input_gen_fns = custom_op_kernel._adapt_user_input_gen_fns(inputs, op_overload, user_input_gen_fns)

    token_decompositions = [_with_autotune_token(decomp) for decomp in decompositions]
    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=token_decompositions,
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
        input_gen_fns=input_gen_fns if input_gen_fns else None,
        config_patches_list=config_patches_list,
    )
    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}.")

    selected_result, winning_choice = autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
        is_collective=custom_op_kernel._detect_collective_ops(choices),
        min_speedup_threshold=min_speedup_threshold,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    if winning_choice is None or winning_choice.gm is None:
        validate_ir(selected_result)
        return selected_result, winning_choice

    operations_before = len(V.graph.operations)
    result = inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
    if winning_choice.config_patches:
        for operation in V.graph.operations[operations_before:]:
            operation.set_config_patches(winning_choice.config_patches.copy())
    validate_ir(result)
    return result, winning_choice


def install_mutation_only_custom_op_autotuning() -> None:
    """Install the mutation-only custom-op autotuning compatibility patch.

    The patch is process-wide because Inductor's lowering registry is
    process-wide.  It is idempotent and delegates every tensor-returning
    custom op to the original implementation unchanged.

    Mutation-only registrations intentionally do not receive Inductor's
    ordinary eager fallback choice: that choice has no output tensor whose
    layout can represent the mutated buffers.  Register a baseline as one of
    the supplied ``CustomOpConfig`` decompositions when one is needed.
    """

    if hasattr(custom_op_kernel, _PATCH_ATTRIBUTE):
        return

    original = custom_op_kernel.autotune_custom_op

    @wraps(original)
    def patched_autotune_custom_op(
        name: str,
        decompositions: list[Callable[..., Any]],
        inputs: list[Any],
        non_tensor_args: list[dict[str, Any]],
        op_overload: Any,
        user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        config_patches_list: list[dict[str, Any]] | None = None,
        min_speedup_threshold: float = 1.0,
        benchmark_with_cudagraphs: bool = False,
    ) -> tuple[Any, Any]:
        if not _is_mutation_only_op(op_overload):
            return original(
                name=name,
                decompositions=decompositions,
                inputs=inputs,
                non_tensor_args=non_tensor_args,
                op_overload=op_overload,
                user_input_gen_fns=user_input_gen_fns,
                config_patches_list=config_patches_list,
                min_speedup_threshold=min_speedup_threshold,
                benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            )

        return _autotune_mutation_only_custom_op(
            name=name,
            decompositions=decompositions,
            inputs=inputs,
            non_tensor_args=non_tensor_args,
            op_overload=op_overload,
            user_input_gen_fns=user_input_gen_fns,
            config_patches_list=config_patches_list,
            min_speedup_threshold=min_speedup_threshold,
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
        )

    setattr(custom_op_kernel, _PATCH_ATTRIBUTE, original)
    setattr(custom_op_kernel, original.__name__, patched_autotune_custom_op)
