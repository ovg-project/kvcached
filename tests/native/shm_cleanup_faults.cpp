// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "mem_info_tracker.hpp"
#include <cassert>
#include <cstdarg>
static int fault = 0;
void now_to_string(char *buf, int length) { snprintf(buf, length, "test"); }
extern "C" int __real_fcntl(int, int, ...);
extern "C" int __wrap_fcntl(int fd, int cmd, ...) {
  va_list args;
  va_start(args, cmd);
  if (cmd == F_DUPFD_CLOEXEC) {
    int arg = va_arg(args, int);
    va_end(args);
    if (fault == 1) {
      errno = EMFILE;
      return -1;
    }
    return __real_fcntl(fd, cmd, arg);
  }
  assert(cmd == F_OFD_SETLK);
  auto *arg = va_arg(args, struct flock *);
  va_end(args);
  if (fault == 2) {
    errno = ENOLCK;
    return -1;
  }
  return __real_fcntl(fd, cmd, arg);
}
extern "C" int __real_fstat(int, struct stat *);
extern "C" int __wrap_fstat(int fd, struct stat *info) {
  if (fault == 3) {
    errno = EIO;
    return -1;
  }
  return __real_fstat(fd, info);
}
extern "C" int __real_unlink(const char *);
extern "C" int __wrap_unlink(const char *path) {
  // The tracker must never invoke unlink, even on failed initialization.
  assert(false);
  return __real_unlink(path);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  for (fault = 1; fault <= 3; ++fault) {
    bool failed = false;
    try {
      kvcached::MemInfoTracker owner(1024, 0, argv[1]);
    } catch (const std::runtime_error &) {
      failed = true;
    }
    assert(failed);
  }
  fault = 0;
  {
    kvcached::MemInfoTracker owner(2048, 0, argv[1]);
    assert(owner.release_segment());
    assert(owner.release_segment());
  }
  assert(access(argv[1], F_OK) == 0);
  assert(__real_unlink(argv[1]) == 0);
  puts("PASS: native startup faults refuse unlocked usage; release never "
       "unlinks");
}
