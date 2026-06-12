#pragma once

#include <torch/headeronly/util/Exception.h>

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
