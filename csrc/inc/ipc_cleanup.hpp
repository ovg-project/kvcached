// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __linux__
#include <arpa/inet.h>
#include <endian.h>
#include <sys/socket.h>
#include <sys/un.h>
#endif

namespace kvcached::ipc_cleanup {

inline bool enabled() {
  const char *mode = std::getenv("KVCACHED_IPC_CLEANUP");
  return mode && std::strcmp(mode, "reaper") == 0;
}

class Fd {
public:
  explicit Fd(int fd = -1) : fd_(fd) {}
  ~Fd() { reset(); }
  Fd(const Fd &) = delete;
  Fd &operator=(const Fd &) = delete;
  int get() const { return fd_; }
  void reset(int fd = -1) {
    if (fd_ >= 0)
      ::close(fd_);
    fd_ = fd;
  }

private:
  int fd_;
};

inline void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(std::string(message) + ": " +
                             std::strerror(errno));
}

inline std::string directory() {
  const char *root = std::getenv("KVCACHED_REAPER_DIR");
  std::string path = root && root[0] ? root
                                     : "/dev/shm/.kvcached-lifecycle-" +
                                           std::to_string(::geteuid());
  require(path[0] == '/', "Reaper directory must be absolute");
  while (path.size() > 1 && path.back() == '/')
    path.pop_back();
  require(::mkdir(path.c_str(), 0700) == 0 || errno == EEXIST,
          "Cannot create reaper directory");
  struct stat info{};
  require(::lstat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode) &&
              info.st_uid == ::geteuid() && !(info.st_mode & 0077),
          "Reaper directory must be private and owned by the current UID");
  return path;
}

inline std::string control_path(const std::string &name) {
  const std::string path =
      name.size() && name[0] == '/' ? name : "/dev/shm/" + name;
  const std::string base =
      path.substr(0, 9) == "/dev/shm/" ? path.substr(9) : "";
  require(!base.empty() && base != "." && base != ".." &&
              base.find('/') == std::string::npos &&
              base.find('\0') == std::string::npos,
          "Reaper mode requires a control file directly in /dev/shm");
  return path;
}

inline unsigned slot(const std::string &path) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char byte : path)
    hash = (hash ^ byte) * UINT64_C(1099511628211);
  return hash % 64;
}

class NameLock {
public:
  NameLock(const std::string &root, const std::string &path)
      : fd_(::open((root + "/gate-" + std::to_string(slot(path))).c_str(),
                   O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600)) {
    require(fd_.get() >= 0, "Cannot open name lock");
    struct stat info{};
    require(::fstat(fd_.get(), &info) == 0 && S_ISREG(info.st_mode) &&
                info.st_uid == ::geteuid(),
            "Unverified name lock");
    require(::flock(fd_.get(), LOCK_EX) == 0, "Cannot acquire name lock");
  }

private:
  Fd fd_;
};

class Lease {
public:
  explicit Lease(const std::string &name) {
#ifdef __linux__
    const std::string path = control_path(name);
    const std::string root = directory();
    NameLock gate(root, path);
    fd_.reset(
        ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0666));
    require(fd_.get() >= 0, "Cannot open control file");
    struct stat info{};
    require(::fstat(fd_.get(), &info) == 0 && S_ISREG(info.st_mode) &&
                info.st_uid == ::geteuid(),
            "Unverified control file");
    require(info.st_size == 24 ||
                (info.st_size == 0 && ::ftruncate(fd_.get(), 24) == 0),
            "Unexpected control-file layout");
    struct flock lock{};
    lock.l_type = F_RDLCK;
    lock.l_whence = SEEK_SET;
    lock.l_len = 1;
    require(::fcntl(fd_.get(), F_OFD_SETLK, &lock) == 0,
            "Cannot establish control-file usage lease");
    registered_ = register_owner(root, path);
#else
    throw std::runtime_error("Reaper mode requires Linux OFD locks");
#endif
  }

  bool registered() const { return registered_; }
  // Close, do not UNLCK: fork children can still hold this description.
  void close() { fd_.reset(); }

private:
#ifdef __linux__
  bool register_owner(const std::string &root, const std::string &path) {
    Fd proof(::open(("/proc/self/fd/" + std::to_string(fd_.get())).c_str(),
                    O_RDWR | O_CLOEXEC));
    struct stat original{};
    struct stat reopened{};
    if (proof.get() < 0 || ::fstat(fd_.get(), &original) < 0 ||
        ::fstat(proof.get(), &reopened) < 0 ||
        original.st_dev != reopened.st_dev ||
        original.st_ino != reopened.st_ino)
      return false;
    Fd connection(::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
    if (connection.get() < 0)
      return false;
    struct timeval timeout{1, 0};
    if (::setsockopt(connection.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout,
                     sizeof(timeout)) < 0 ||
        ::setsockopt(connection.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout,
                     sizeof(timeout)) < 0)
      return false;
    struct sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const std::string endpoint = root + "/reaper.sock";
    if (endpoint.size() >= sizeof(address.sun_path)) {
      errno = ENAMETOOLONG;
      return false;
    }
    std::memcpy(address.sun_path, endpoint.c_str(), endpoint.size() + 1);
    if (::connect(connection.get(), reinterpret_cast<sockaddr *>(&address),
                  sizeof(address)) < 0)
      return false;
    struct ucred credentials{};
    socklen_t length = sizeof(credentials);
    if (::getsockopt(connection.get(), SOL_SOCKET, SO_PEERCRED, &credentials,
                     &length) < 0 ||
        credentials.uid != ::geteuid())
      return false;
    uint32_t version = htonl(1);
    struct iovec data[] = {{&version, sizeof(version)},
                           {const_cast<char *>(path.data()), path.size()}};
    alignas(cmsghdr) char ancillary[CMSG_SPACE(sizeof(int))]{};
    struct msghdr message{};
    message.msg_iov = data;
    message.msg_iovlen = 2;
    message.msg_control = ancillary;
    message.msg_controllen = sizeof(ancillary);
    auto *header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    const int proof_fd = proof.get();
    std::memcpy(CMSG_DATA(header), &proof_fd, sizeof(proof_fd));
    if (::sendmsg(connection.get(), &message, MSG_NOSIGNAL) !=
        static_cast<ssize_t>(sizeof(version) + path.size()))
      return false;
    unsigned char ack[17]{};
    if (::recv(connection.get(), ack, sizeof(ack), MSG_TRUNC) != sizeof(ack) ||
        ack[0] != 1)
      return false;
    uint64_t device, inode;
    std::memcpy(&device, ack + 1, sizeof(device));
    std::memcpy(&inode, ack + 9, sizeof(inode));
    return be64toh(device) == static_cast<uint64_t>(original.st_dev) &&
           be64toh(inode) == static_cast<uint64_t>(original.st_ino);
  }
#endif
  Fd fd_;
  bool registered_ = false;
};

} // namespace kvcached::ipc_cleanup
