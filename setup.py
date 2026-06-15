import os
import shlex
import shutil
import subprocess
from typing import Any

import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


def _quote_command_arg(arg: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([arg])
    return shlex.quote(arg)


def _prepend_ccache(command: str, ccache_path: str) -> str:
    if "ccache" in command.lower():
        return command
    return f"{_quote_command_arg(ccache_path)} {command}"


def _enable_ccache() -> None:
    ccache_path = shutil.which("ccache")
    if not ccache_path:
        return

    assert CUDA_HOME is not None
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
    os.environ["PYTORCH_NVCC"] = _prepend_ccache(os.getenv("PYTORCH_NVCC", _quote_command_arg(nvcc)), ccache_path)

    original_write_ninja_file = getattr(cpp_extension, "_write_ninja_file")
    if original_write_ninja_file is None:
        return

    if getattr(original_write_ninja_file, "_sageattention_ccache", False):
        return

    ccache_prefix = "" if "ccache" in os.getenv("CXX", "").lower() else f"{_quote_command_arg(ccache_path)} "

    def write_ninja_file_with_ccache(*args: Any, **kwargs: Any) -> None:
        original_write_ninja_file(*args, **kwargs)
        path = kwargs.get("path", args[0] if args else None)
        if path is None or not os.path.exists(path):
            return

        with open(path, encoding="utf-8") as f:
            content = f.read()

        patched = content.replace(
            "  command = cl /showIncludes",
            f"  command = {ccache_prefix}$cxx /showIncludes",
        ).replace(
            "  command = $cxx -MMD",
            f"  command = {ccache_prefix}$cxx -MMD",
        )

        if patched != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)

    setattr(write_ninja_file_with_ccache, "_sageattention_ccache", True)
    setattr(cpp_extension, "_write_ninja_file", write_ninja_file_with_ccache)


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


build_triton_only = _env_flag_enabled("SAGEATTN_BUILD_TRITON_ONLY")

if not build_triton_only:
    if CUDA_HOME is None:
        raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available to build SageAttention.")
    _enable_ccache()

if os.name == "nt":
    cxx_flags = ["/O2", "/openmp", "/std:c++17", "/permissive-", "-DENABLE_BF16"]
else:
    cxx_flags = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]

cxx_flags += ["-DPy_LIMITED_API=0x030A0000", "-DTORCH_STABLE_ONLY"]

nvcc_flags = [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    f"--threads={os.cpu_count()}",
    "-diag-suppress=174",
    "-diag-suppress=177",
    "-diag-suppress=221",
    "-DPy_LIMITED_API=0x030A0000",
    "-DTORCH_STABLE_ONLY",
]

if os.name == "nt":
    nvcc_flags += ["-Xcompiler=/Zc:preprocessor", "-D_WIN32=1", "-DUSE_CUDA=1"]
else:
    abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    cxx_flags += [f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
    nvcc_flags += [f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]

cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
if cxx_append:
    cxx_flags += cxx_append.split()

nvcc_append = os.getenv("NVCC_APPEND_FLAGS", "").strip()
if nvcc_append:
    nvcc_flags += nvcc_append.split()


def _gencode(arch: int) -> list[str]:
    return ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]


def _cuda_toolkit_version() -> tuple[int, int]:
    """(major, minor) of the nvcc/toolkit used to build, falling back to torch's."""
    raw = getattr(cpp_extension, "_get_cuda_arch_flags", None)  # noqa: F841 (touch to ensure import)
    version = None
    if CUDA_HOME is not None:
        nvcc = os.path.join(CUDA_HOME, "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
        try:
            out = subprocess.check_output([nvcc, "--version"], text=True)
            for token in out.replace(",", " ").split():
                if token.startswith("V") and token[1:2].isdigit():
                    parts = token[1:].split(".")
                    version = (int(parts[0]), int(parts[1]))
                    break
        except Exception:
            version = None
    if version is None and torch.version.cuda:
        major, minor = torch.version.cuda.split(".")[:2]
        version = (int(major), int(minor))
    return version or (0, 0)


def _resolve_sm89_arches() -> list[int]:
    """Arches for the fp8 sm89 extension: env override or auto-detect.

    SAGEATTN_CUDA_ARCH may be e.g. "89", "120", or "89;120" (also accepts
    commas / "8.9"). Otherwise we detect the installed GPU's capability.
    Only sm_89 and sm_120 are supported by the fp8 kernels.
    """
    supported = {89, 120}
    override = os.getenv("SAGEATTN_CUDA_ARCH", "").strip()
    if override:
        arches = []
        for tok in override.replace(",", ";").split(";"):
            tok = tok.strip().replace(".", "")
            if tok:
                arches.append(int(tok))
    else:
        arches = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                arches.append(major * 10 + minor)
        arches = arches or [89]
    arches = sorted({a for a in arches if a in supported})
    if not arches:
        raise RuntimeError(
            "No supported fp8 arch resolved for _qattn_sm89 (need sm_89 or sm_120). "
            "Set SAGEATTN_CUDA_ARCH=89 and/or 120."
        )
    # Guard against toolkits too old for the requested arch.
    cu_major, cu_minor = _cuda_toolkit_version()
    cu = cu_major * 100 + cu_minor
    if 89 in arches and cu < 1204:
        raise RuntimeError(f"sm_89 fp8 kernels need CUDA >= 12.4 (found {cu_major}.{cu_minor}).")
    if 120 in arches and cu < 1208:
        raise RuntimeError(f"sm_120 fp8 kernels need CUDA >= 12.8 (found {cu_major}.{cu_minor}).")
    return arches


def _cccl_include_flags() -> list[str]:
    """conda/micromamba CUDA layouts put <cuda/pipeline> under a cccl subdir."""
    if CUDA_HOME is None:
        return []
    candidates = [
        os.path.join(CUDA_HOME, "include", "cccl"),
        os.path.join(CUDA_HOME, "include", "targets", "x64", "cccl"),
    ]
    flags = []
    for path in candidates:
        if os.path.isdir(path):
            flags += ["-I", path]
    return flags


if build_triton_only:
    ext_modules = []
else:
    sm89_arches = _resolve_sm89_arches()
    # Build the fp16 (sm80) extension for sm_80 plus the detected arch(es) so the
    # fp16 fallback also runs on Ada/Blackwell (sm_80 SASS is not forward
    # compatible across compute-capability majors).
    sm80_nvcc_flags = list(nvcc_flags) + _gencode(80)
    for arch in sm89_arches:
        if arch != 80:
            sm80_nvcc_flags += _gencode(arch)
    sm89_nvcc_flags = nvcc_flags + _cccl_include_flags()
    for arch in sm89_arches:
        sm89_nvcc_flags += _gencode(arch)

    ext_modules = [
        CUDAExtension(
            name="sageattention._qattn_sm80",
            sources=[
                "csrc/qattn/pybind_sm80.cpp",
                "csrc/qattn/qk_int8_sv_f16_accum_f16_attn.cu",
                "csrc/qattn/qk_int8_sv_f16_accum_f16_attn_inst_buf.cu",
                "csrc/qattn/qk_int8_sv_f16_accum_f16_fuse_v_mean_attn.cu",
                "csrc/qattn/qk_int8_sv_f16_accum_f32_attn.cu",
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": sm80_nvcc_flags},
            py_limited_api=True,
        ),
        CUDAExtension(
            name="sageattention._qattn_sm89",
            sources=[
                "csrc/qattn/pybind_sm89.cpp",
                "csrc/qattn/qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
                "csrc/qattn/qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": sm89_nvcc_flags},
            py_limited_api=True,
        ),
    ]

max_jobs = os.getenv("EXT_PARALLEL", os.getenv("MAX_JOBS", str(os.cpu_count() or 1)))
os.environ.setdefault("MAX_JOBS", max_jobs)

setup(
    name="sageattention",
    version="2.2.0" + os.environ.get("SAGEATTENTION_WHEEL_VERSION_SUFFIX", ""),
    author="SageAttention team",
    license="Apache 2.0 License",
    packages=find_packages(),
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
