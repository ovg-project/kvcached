// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "ipc_cleanup.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  kvcached::ipc_cleanup::Lease lease(argv[1]);
  std::cout << "ready " << lease.registered() << " "
            << kvcached::ipc_cleanup::slot(argv[1]) << std::endl;
  std::string command;
  while (std::getline(std::cin, command)) {
    if (command == "close") {
      lease.close();
      std::cout << "closed" << std::endl;
    } else if (command == "crash") {
      ::_exit(0);
    } else if (command == "exit") {
      return 0;
    }
  }
}
