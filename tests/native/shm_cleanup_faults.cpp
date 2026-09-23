// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "mem_info_tracker.hpp"
#include <cassert>
#include <cstdarg>
#include <memory>

static bool fail_dup = false;
static bool fail_stat = false;
static bool fail_fstat = false;
static bool fail_unlink = false;
static int hits = 0;

void now_to_string(char *buf, int length) { snprintf(buf, length, "test"); }

extern "C" int __real_fcntl(int, int, ...);
extern "C" int __wrap_fcntl(int fd, int command, ...) {
  assert(command == F_DUPFD_CLOEXEC);
  if (fail_dup) {
    ++hits;
    errno = EMFILE;
    return -1;
  }
  va_list args;
  va_start(args, command);
  const int argument = va_arg(args, int);
  va_end(args);
  return __real_fcntl(fd, command, argument);
}

extern "C" int __real_stat(const char *, struct stat *);
extern "C" int __wrap_stat(const char *path, struct stat *info) {
  if (fail_stat) {
    ++hits;
    errno = EACCES;
    return -1;
  }
  return __real_stat(path, info);
}

extern "C" int __real_fstat(int, struct stat *);
extern "C" int __wrap_fstat(int fd, struct stat *info) {
  if (fail_fstat) {
    ++hits;
    errno = EIO;
    return -1;
  }
  return __real_fstat(fd, info);
}

extern "C" int __real_unlink(const char *);
extern "C" int __wrap_unlink(const char *path) {
  if (fail_unlink) {
    ++hits;
    errno = EACCES;
    return -1;
  }
  return __real_unlink(path);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const std::string path = argv[1];
  assert(access(path.c_str(), F_OK) != 0);
  fail_dup = true;
  auto unknown = std::make_unique<kvcached::MemInfoTracker>(1024, 0, path);
  fail_dup = false;
  assert(!unknown->release_segment());
  unknown.reset();
  assert(access(path.c_str(), F_OK) == 0);
  assert(unlink(path.c_str()) == 0);

  for (bool *fault : {&fail_stat, &fail_fstat, &fail_unlink}) {
    auto owner = std::make_unique<kvcached::MemInfoTracker>(1024, 0, path);
    *fault = true;
    assert(!owner->release_segment());
    *fault = false;
    assert(access(path.c_str(), F_OK) == 0);
    assert(owner->release_segment());
    assert(access(path.c_str(), F_OK) != 0);
    auto replacement =
        std::make_unique<kvcached::MemInfoTracker>(2048, 0, path);
    owner.reset();
    assert(access(path.c_str(), F_OK) == 0);
    assert(replacement->release_segment());
  }

  // A pool created during an unlink retry must join the still-live identity.
  auto first = std::make_unique<kvcached::MemInfoTracker>(1024, 0, path);
  fail_unlink = true;
  assert(!first->release_segment());
  fail_unlink = false;
  auto second = std::make_unique<kvcached::MemInfoTracker>(2048, 1, path);
  assert(first->release_segment());
  first.reset();
  assert(access(path.c_str(), F_OK) == 0);
  assert(second->release_segment());
  assert(access(path.c_str(), F_OK) != 0);

  auto known = std::make_unique<kvcached::MemInfoTracker>(1024, 0, path);
  fail_dup = true;
  auto live_unknown = std::make_unique<kvcached::MemInfoTracker>(2048, 1, path);
  fail_dup = false;
  assert(!known->release_segment());
  assert(access(path.c_str(), F_OK) == 0);
  assert(!live_unknown->release_segment());
  assert(known->release_segment());
  assert(access(path.c_str(), F_OK) != 0);
  assert(hits == 6);
  printf("PASS: 6 native syscall faults, safe retries and shared ownership\n");
}
