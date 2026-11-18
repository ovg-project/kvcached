// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kvcached {
namespace control_segment {
// The first 24 bytes retain the MemInfoStruct ABI. The trailer identifies the
// usage-lock protocol, not a proof that unmodified clients have stopped. Before
// upgrading, stop ALL users of this segment (including forked descendants) and
// upgrade its writers/deleters together. Other segments can upgrade separately.
constexpr off_t marker_offset = 24;
constexpr char marker[] = "KVCOFD01";
constexpr size_t marker_size = sizeof(marker) - 1;
constexpr size_t size = marker_offset + marker_size;

inline bool retain(int fd) {
#if defined(__linux__) && defined(F_OFD_SETLK)
  struct flock lock{};
  lock.l_type = F_RDLCK;
  lock.l_whence = SEEK_SET;
  lock.l_len = 1;
  int result;
  do {
    result = ::fcntl(fd, F_OFD_SETLK, &lock);
  } while (result < 0 && errno == EINTR);
  return result == 0;
#elif defined(__linux__)
  errno = ENOTSUP;
  return false;
#else
  // No protocol marker on platforms without Linux OFD semantics. Keep legacy
  // visibility and refuse safe deletion; do not substitute process locks.
  (void)fd;
  return true;
#endif
}

inline bool publish(int fd) {
#if defined(__linux__) && defined(F_OFD_SETLK)
  return ::pwrite(fd, marker, marker_size, marker_offset) ==
             static_cast<ssize_t>(marker_size) &&
         ::ftruncate(fd, size) == 0;
#else
  (void)fd;
  return true;
#endif
}

inline bool is_current(int fd, const char *path) {
  struct stat opened{}, current{};
  return ::fstat(fd, &opened) == 0 && S_ISREG(opened.st_mode) &&
         ::lstat(path, &current) == 0 && S_ISREG(current.st_mode) &&
         opened.st_dev == current.st_dev && opened.st_ino == current.st_ino;
}
} // namespace control_segment
} // namespace kvcached
