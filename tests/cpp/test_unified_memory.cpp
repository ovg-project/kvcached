// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <sstream>

#include "unified_memory.hpp"

int main() {
  std::istringstream meminfo("MemTotal:           1000 kB\n"
                             "MemFree:             100 kB\n"
                             "MemAvailable:        400 kB\n"
                             "HugePages_Total:       0\n");
  assert(kvcached::parse_mem_available(meminfo) == 400 * 1024);

  std::istringstream missing("MemTotal:           1000 kB\n");
  assert(!kvcached::parse_mem_available(missing));
  return 0;
}
