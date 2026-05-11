// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef enum {
  FATAL = 0,
  ERROR = 1,
  WARNING = 2,
  INFO = 3,
  DEBUG = 4,
  VERBOSE = 5,
} log_level_enum_t;

extern void now_to_string(char *buf, int length);
#ifdef __cplusplus
__attribute__((unused)) static char *logger_level_str[] = {
    (char *)"FATAL", (char *)"ERROR", (char *)"WARNING",
    (char *)"INFO",  (char *)"DEBUG", (char *)"VERBOSE"};
#else
__attribute__((unused)) static char *logger_level_str[] = {
    "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "VERBOSE"};
#endif

// glibc >= 2.30 provides a native gettid() wrapper; only define our own
// syscall-based version on older systems to avoid macro/function conflicts.
#if !defined(__GLIBC__) || !defined(__GLIBC_MINOR__) || (__GLIBC__ < 2) ||     \
    (__GLIBC__ == 2 && __GLIBC_MINOR__ < 30)
#ifndef SYS_gettid
#error "SYS_gettid unavailable on this system"
#endif
static inline pid_t gettid(void) { return (pid_t)syscall(SYS_gettid); }
#endif

#define LOGGER(level, format, ...)                                             \
  ({                                                                           \
    char *_print_level_str = getenv("KVCACHED_LOG_LEVEL");                     \
    char time[64];                                                             \
    now_to_string(time, 64);                                                   \
    int _print_level = 0;                                                      \
    if (_print_level_str == NULL) {                                            \
      _print_level = WARNING;                                                  \
    } else if (_print_level_str[0] == 'F') {                                   \
      _print_level = FATAL;                                                    \
    } else if (_print_level_str[0] == 'E') {                                   \
      _print_level = ERROR;                                                    \
    } else if (_print_level_str[0] == 'W') {                                   \
      _print_level = WARNING;                                                  \
    } else if (_print_level_str[0] == 'I') {                                   \
      _print_level = INFO;                                                     \
    } else if (_print_level_str[0] == 'D') {                                   \
      _print_level = DEBUG;                                                    \
    } else if (_print_level_str[0] == 'V') {                                   \
      _print_level = VERBOSE;                                                  \
    }                                                                          \
    if (level <= _print_level) {                                               \
      fprintf(stderr,                                                          \
              "[KVCACHED_MEMORY_POOL][%s][%s]%s:%d [p:%u t:%u]" format "\n",   \
              logger_level_str[level], time, __FILE__, __LINE__,               \
              (unsigned int)getpid(), (unsigned int)gettid(), ##__VA_ARGS__);  \
    }                                                                          \
    if (level == FATAL) {                                                      \
      exit(-1);                                                                \
    }                                                                          \
  })

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
