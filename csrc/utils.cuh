#pragma once

#include <cuda_runtime_api.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/util/Exception.h>

inline torch::stable::accelerator::DeviceGuard make_device_guard(const torch::stable::Tensor &tensor) {
  return torch::stable::accelerator::DeviceGuard(tensor.get_device_index());
}

inline cudaStream_t get_current_cuda_stream(const torch::stable::Tensor &tensor) {
  // We rely on the raw shim API to get the current CUDA stream.
  // This will be improved in a future release of PyTorch, see https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
  // This is needed because torch.compile may launch kernels on non-default streams.
  // Use this while a tensor-device guard is alive for multi-GPU correctness.
  const auto device_index = tensor.get_device_index();
  void *stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}

#define CHECK_CUDA(x) \
  STD_TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")

#define CHECK_DTYPE(x, true_dtype) \
  STD_TORCH_CHECK(x.scalar_type() == true_dtype, "Tensor " #x " must have dtype (" #true_dtype ")")

#define CHECK_DIMS(x, true_dim) \
  STD_TORCH_CHECK(x.dim() == true_dim, "Tensor " #x " must have dimension number (" #true_dim ")")

#define CHECK_SHAPE(x, ...) \
  STD_TORCH_CHECK((x).sizes().equals({__VA_ARGS__}), "Tensor " #x " must have shape (" #__VA_ARGS__ ")")

#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")

#define CHECK_LASTDIM_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.stride(-1) == 1, "Tensor " #x " must be contiguous at the last dimension")
