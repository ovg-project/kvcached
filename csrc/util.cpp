// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <iostream>
#include <sys/time.h>
#include <time.h>

#define NSEC_PER_USEC (1000ULL)
#define USEC_PER_SEC (1000000ULL)

uint64_t timespec_to_us(struct timespec ts) {
  return (ts.tv_sec * USEC_PER_SEC + ts.tv_nsec / NSEC_PER_USEC);
}

uint64_t get_current_timestamp_in_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return timespec_to_us(ts);
}

void now_to_string(char *buf, int length) {
  auto now = std::chrono::system_clock::now();
  auto seconds = std::chrono::system_clock::to_time_t(now);
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                now.time_since_epoch()) %
            1000000;

  std::tm tm_struct = {};

#ifdef _WIN32
  localtime_s(&tm_struct, &seconds);
#else
  localtime_r(&seconds, &tm_struct);
#endif

  char buffer[64];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_struct);

  std::snprintf(buf, length, "%s.%06ld", buffer, static_cast<long>(us.count()));
}