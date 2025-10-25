// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

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
      LOGE(__VA_ARGS__);                                                       \
    }                                                                          \
  }

#define DRV_CALL(call)                                                         \
  {                                                                            \
    CUresult result = (call);                                                  \
    if (CUDA_SUCCESS != result) {                                              \
      const char *errMsg;                                                      \
      cuGetErrorString(result, &errMsg);                                       \
      ASSERT(0, "Error when exec " #call " %s-%d code:%d err:%s",              \
             __FUNCTION__, __LINE__, result, errMsg);                          \
    }                                                                          \
  }

#define DRV_CALL_RET(call, status_val)                                         \
  {                                                                            \
    CUresult result = (call);                                                  \
    if (CUDA_SUCCESS != result) {                                              \
      const char *errMsg;                                                      \
      cuGetErrorString(result, &errMsg);                                       \
      WARN(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__,  \
           __LINE__, result, errMsg);                                          \
    }                                                                          \
    status_val = result;                                                       \
  }

static inline void checkRtError(cudaError_t res, const char *tok,
                                const char *file, unsigned line) {
  if (res != cudaSuccess) {
    std::cerr << file << ':' << line << ' ' << tok
              << " failed in CUDA runtime (" << (unsigned)res
              << "): " << cudaGetErrorString(res) << std::endl;
    abort();
  }
}

#define CHECK_RT(x) checkRtError(x, #x, __FILE__, __LINE__)

static inline void checkDrvError(CUresult res, const char *tok,
                                 const char *file, unsigned line) {
  if (res != CUDA_SUCCESS) {
    const char *errStr = nullptr;
    (void)cuGetErrorString(res, &errStr);
    std::cerr << file << ':' << line << ' ' << tok << " failed in CUDA driver ("
              << (unsigned)res << "): " << errStr << std::endl;
    abort();
  }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__)
