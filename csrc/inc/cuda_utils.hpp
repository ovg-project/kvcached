// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdio>
#include <iostream>

#include "gpu_vmm.hpp"

#define LOGE(format, ...)                                                      \
  fprintf(stderr, "ERROR: %s:%d: " format "\n", __FILE__, __LINE__,            \
          ##__VA_ARGS__);                                                      \
  fflush(stderr);

#define LOGW(format, ...)                                                      \
  fprintf(stderr, "WARNING: %s:%d: " format "\n", __FILE__, __LINE__,          \
          ##__VA_ARGS__);                                                      \
  fflush(stderr);

#define ASSERT(cond, ...)                                                      \
  {                                                                            \
    if (!(cond)) {                                                             \
      LOGE(__VA_ARGS__);                                                       \
      assert(0);                                                               \
    }                                                                          \
  }

#define WARN(cond, ...)                                                        \
  {                                                                            \
    if (!(cond)) {                                                             \
      LOGW(__VA_ARGS__);                                                       \
    }                                                                          \
  }

#define DRV_CALL(call) CHECK_GPU(call)

#define DRV_CALL_RET(call, status_val)                                         \
  {                                                                            \
    auto result = (call);                                                      \
    if (!kvcached::gpu_vmm::is_success(result)) {                              \
      WARN(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__,  \
           __LINE__, static_cast<int>(result),                                 \
           kvcached::gpu_vmm::error_string(result));                           \
    }                                                                          \
    status_val = result;                                                       \
  }

#define CHECK_GPU(x) kvcached::gpu_vmm::check((x), #x, __FILE__, __LINE__)
#define CHECK_RT(x) CHECK_GPU(x)
#define CHECK_DRV(x) CHECK_GPU(x)
