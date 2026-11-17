// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>
#include <fstream>
#include <istream>
#include <optional>
#include <string>

namespace kvcached {

// On a unified-memory GPU such as GB10, device memory is system memory, and
// the driver's free count leaves out page cache that the kernel reclaims when
// a device allocation needs it. MemAvailable includes it; vLLM and SGLang size
// their KV cache from it on these devices too.

inline std::optional<size_t> parse_mem_available(std::istream &meminfo) {
  std::string line;
  while (std::getline(meminfo, line)) {
    unsigned long long kib = 0;
    if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kib) == 1) {
      return static_cast<size_t>(kib) * 1024;
    }
  }
  return std::nullopt;
}

inline std::optional<size_t> read_mem_available() {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo) {
    return std::nullopt;
  }
  return parse_mem_available(meminfo);
}

} // namespace kvcached
