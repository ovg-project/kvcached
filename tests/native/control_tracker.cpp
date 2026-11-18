// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
// A real MemInfoTracker process for Python/C++ interoperability tests. No CUDA
// allocation is made; compiling the header only needs the installed toolkit.
#include "mem_info_tracker.hpp"
#include <iostream>
#include <string>
void now_to_string(char *buf, int length) { snprintf(buf, length, "test"); }
int main(int argc, char **argv) {
  if (argc != 2 && argc != 3)
    return 2;
  if (argc == 3) {
    std::cout << "waiting" << std::endl;
    std::string start;
    std::getline(std::cin, start);
  }
  try {
    kvcached::MemInfoTracker owner(4096, 0, argv[1]);
    std::cout << "ready" << std::endl;
    std::string line;
    std::getline(std::cin, line);
    owner.release_segment();
    std::cout << "closed" << std::endl;
    std::getline(std::cin, line);
    return 0;
  } catch (const std::runtime_error &) {
    std::cout << "failed" << std::endl;
    return 3;
  }
}
