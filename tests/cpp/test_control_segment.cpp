// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "control_segment.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
int main() {
#if defined(__linux__) && defined(F_OFD_SETLK)
  char path[] = "/tmp/kvcached-control-XXXXXX";
  int holder = mkstemp(path);
  assert(holder >= 0 && kvcached::control_segment::retain(holder));
  assert(kvcached::control_segment::publish(holder));
  int other = open(path, O_RDWR);
  struct flock request{};
  request.l_type = F_WRLCK;
  request.l_whence = SEEK_SET;
  request.l_len = 1;
  assert(fcntl(other, F_OFD_SETLK, &request) == -1 && errno == EAGAIN);
  int duplicate = dup(holder);
  close(holder);
  assert(fcntl(other, F_OFD_SETLK, &request) == -1 && errno == EAGAIN);
  close(duplicate);
  assert(fcntl(other, F_OFD_SETLK, &request) == 0);
  assert(kvcached::control_segment::is_current(other, path));
  assert(unlink(path) == 0);
  assert(!kvcached::control_segment::is_current(other, path));
  close(other);
  puts("control segment OFD/dup/identity passed");
#endif
}
