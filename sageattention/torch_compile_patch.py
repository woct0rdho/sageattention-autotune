import logging
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor.codegen.subgraph import SubgraphChoiceCaller
from torch._inductor.select_algorithm import AlgorithmSelectorCache, get_algorithm_selector_cache

TimingTarget = Callable[[Any, tuple[torch.Tensor, ...], torch.Tensor], float]
SelectionCallback = Callable[[Any], None]

_PATCH_ATTRIBUTE = "_sageattention_custom_timing_patch"


class _CustomTimingPatch:
    def __init__(self) -> None:
        self.timing_targets: dict[str, TimingTarget] = {}
        self.selection_callbacks: dict[str, SelectionCallback] = {}
        self.original_subgraph_benchmark: Callable[..., float] | None = None
        self.original_subgraph_output_node: Callable[..., Any] | None = None
        self.original_benchmark_in_sub_process: Callable[..., dict[Any, float]] | None = None
        self.installed = False
        self.choice_preprocessor_installed = False

    def register(
        self,
        name_prefix: str,
        target: TimingTarget,
        on_select: SelectionCallback | None,
    ) -> None:
        self.install()
        self.install_choice_preprocessor()
        self.timing_targets[name_prefix] = target
        if on_select is not None:
            self.selection_callbacks[name_prefix] = on_select

    def install(self) -> None:
        if self.installed:
            return

        self.original_subgraph_benchmark = SubgraphChoiceCaller.benchmark
        self.original_subgraph_output_node = SubgraphChoiceCaller.output_node
        self.original_benchmark_in_sub_process = AlgorithmSelectorCache.benchmark_in_sub_process

        patch = self

        def benchmark(self: Any, *args: Any, out: torch.Tensor) -> float:
            target = patch.timing_target_for(self)
            if target is None:
                assert patch.original_subgraph_benchmark is not None
                return patch.original_subgraph_benchmark(self, *args, out=out)

            if self._benchmark_with_cudagraphs:
                raise NotImplementedError("custom timing targets do not support CUDA graph benchmarking")

            return target(self, args, out)

        def output_node(self: Any) -> Any:
            for name_prefix, callback in patch.selection_callbacks.items():
                if getattr(self, "name", "").startswith(name_prefix):
                    callback(self)
                    break
            assert patch.original_subgraph_output_node is not None
            return patch.original_subgraph_output_node(self)

        @classmethod
        def benchmark_in_sub_process(
            cls: type[Any],
            choices: Any,
            input_nodes: Any,
            layout: Any,
            input_gen_fns: Any,
            hint_override: int | None = None,
        ) -> dict[Any, float]:
            custom_choices = [choice for choice in choices if patch.timing_target_for(choice) is not None]
            if not custom_choices:
                assert patch.original_benchmark_in_sub_process is not None
                return patch.original_benchmark_in_sub_process(
                    choices,
                    input_nodes,
                    layout,
                    input_gen_fns,
                    hint_override=hint_override,
                )

            # Python timing callbacks cannot be serialized into Inductor's autotune
            # subprocess protocol, so only those choices stay in the current process.
            other_choices = [choice for choice in choices if choice not in custom_choices]
            timings = {}
            if other_choices:
                assert patch.original_benchmark_in_sub_process is not None
                timings.update(
                    patch.original_benchmark_in_sub_process(
                        other_choices,
                        input_nodes,
                        layout,
                        input_gen_fns,
                        hint_override=hint_override,
                    )
                )
            timings.update(
                cls.benchmark_in_current_process(
                    custom_choices,
                    input_nodes,
                    layout,
                    input_gen_fns,
                    hint_override=hint_override,
                )
            )
            return timings

        SubgraphChoiceCaller.benchmark = benchmark
        SubgraphChoiceCaller.output_node = output_node
        # This is an intentional classmethod monkey patch. The exact torch internal
        # signature is version-specific and too narrow for a reusable wrapper.
        setattr(AlgorithmSelectorCache, "benchmark_in_sub_process", benchmark_in_sub_process)
        self.installed = True

    def install_choice_preprocessor(self) -> None:
        if self.choice_preprocessor_installed:
            return

        def custom_timing_choices_only(choices: list[Any]) -> list[Any]:
            custom_choices = [choice for choice in choices if self.timing_target_for(choice) is not None]
            if not custom_choices:
                return choices

            # Custom-op fallback choices use normal forward-only timing, which is not
            # comparable to a caller-provided forward + backward timing target.
            dropped_count = len(choices) - len(custom_choices)
            if dropped_count:
                logging.getLogger(__name__).debug("Dropped %d fallback choices for custom timing target", dropped_count)
            return custom_choices

        get_algorithm_selector_cache().add_preprocessing_fn(custom_timing_choices_only)
        self.choice_preprocessor_installed = True

    def timing_target_for(self, choice: Any) -> TimingTarget | None:
        name = getattr(choice, "name", "")
        for name_prefix, target in self.timing_targets.items():
            if name.startswith(name_prefix):
                return target
        return None


# The monkey patch itself must be process-wide. Store its registry on the torch
# class being patched instead of keeping mutable module-level globals here.
def _custom_timing_patch() -> _CustomTimingPatch:
    patch = getattr(SubgraphChoiceCaller, _PATCH_ATTRIBUTE, None)
    if patch is None:
        patch = _CustomTimingPatch()
        setattr(SubgraphChoiceCaller, _PATCH_ATTRIBUTE, patch)
    return patch


def register_custom_timing_target(
    name_prefix: str,
    target: TimingTarget,
    on_select: SelectionCallback | None = None,
) -> None:
    _custom_timing_patch().register(name_prefix, target, on_select)
