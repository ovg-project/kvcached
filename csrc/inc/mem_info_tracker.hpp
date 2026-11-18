// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "control_segment.hpp"
#include "gpu_utils.hpp"

namespace kvcached {

static constexpr const char *SHM_DIR = "/dev/shm";

// Memory info struct stored in shared memory, compatible with Python
// MemInfoStruct. Layout: [total_size(int64), used_size(int64),
// prealloc_size(int64)]
struct MemInfoStruct {
  int64_t total_size;
  int64_t used_size;
  int64_t prealloc_size;

  static constexpr int N_FIELDS = 3;
  static constexpr size_t SHM_SIZE = sizeof(int64_t) * N_FIELDS;

  MemInfoStruct() : total_size(0), used_size(0), prealloc_size(0) {}
  MemInfoStruct(int64_t total, int64_t used, int64_t prealloc)
      : total_size(total), used_size(used), prealloc_size(prealloc) {}
};

// RAII class for file-lock + mmap operations on /dev/shm files
class RwLockedShm {
public:
  enum LockType { RLOCK = LOCK_SH, WLOCK = LOCK_EX };

  RwLockedShm(const std::string &file_path, size_t size, LockType lock_type,
              bool retain_usage = false)
      : file_path_(get_ipc_path(file_path)), size_(size), lock_type_(lock_type),
        fd_(-1), mapped_(nullptr), retain_usage_(retain_usage) {}

  ~RwLockedShm() { close(); }

  // Open and lock the shared memory file, returns whether successful
  bool open() {
    // Usage is acquired before the short data flock. A concurrent delete can
    // unlink an fd opened before its exclusive lock; recheck the path before
    // sizing or writing. Only initialization pays for the lifetime protocol.
    for (int attempt = 0; attempt < 4; ++attempt) {
      int flags = O_RDWR | O_CLOEXEC;
      if (lock_type_ == WLOCK)
        flags |= O_CREAT;
      if (retain_usage_)
        flags |= O_NOFOLLOW;
      fd_ = ::open(file_path_.c_str(), flags, 0666);
      if (fd_ < 0)
        return false;
      if (retain_usage_ && !control_segment::retain(fd_)) {
        const int error = errno;
        ::close(fd_);
        fd_ = -1;
        errno = error;
        // Contention with a requested deletion is recoverable at startup;
        // never run as an unlocked participant after any other failure.
        if (error == EAGAIN || error == EACCES)
          continue;
        return false;
      }
      if (flock(fd_, lock_type_) < 0) {
        close();
        return false;
      }
      if (retain_usage_ &&
          !control_segment::is_current(fd_, file_path_.c_str())) {
        close();
        errno = ESTALE;
        continue;
      }
      struct stat st{};
      if (fstat(fd_, &st) < 0 ||
          (static_cast<size_t>(st.st_size) < size_ &&
           (lock_type_ != WLOCK || ftruncate(fd_, size_) < 0))) {
        close();
        return false;
      }
      break;
    }
    if (fd_ < 0)
      return false;

    // mmap
    int prot = (lock_type_ == WLOCK) ? (PROT_READ | PROT_WRITE) : PROT_READ;
    mapped_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      mapped_ = nullptr;
      flock(fd_, LOCK_UN);
      ::close(fd_);
      fd_ = -1;
      return false;
    }

    return true;
  }

  void close() {
    if (mapped_ != nullptr) {
      munmap(mapped_, size_);
      mapped_ = nullptr;
    }
    if (fd_ >= 0) {
      flock(fd_, LOCK_UN);
      ::close(fd_);
      fd_ = -1;
    }
  }

  void *data() { return mapped_; }
  const void *data() const { return mapped_; }

  const std::string &path() const { return file_path_; }
  int duplicate_fd() const { return ::fcntl(fd_, F_DUPFD_CLOEXEC, 0); }

  // Read MemInfoStruct from mmap buffer
  MemInfoStruct read_mem_info() const {
    MemInfoStruct info;
    if (mapped_) {
      const int64_t *arr = static_cast<const int64_t *>(mapped_);
      info.total_size = arr[0];
      info.used_size = arr[1];
      info.prealloc_size = arr[2];
    }
    return info;
  }

  // Write MemInfoStruct to mmap buffer
  void write_mem_info(const MemInfoStruct &info) {
    if (mapped_) {
      int64_t *arr = static_cast<int64_t *>(mapped_);
      arr[0] = info.total_size;
      arr[1] = info.used_size;
      arr[2] = info.prealloc_size;
    }
  }

private:
  static std::string get_ipc_path(const std::string &name) {
    if (name.empty())
      return "";
    if (name[0] == '/')
      return name;
    return std::string(SHM_DIR) + "/" + name;
  }

  std::string file_path_;
  size_t size_;
  LockType lock_type_;
  int fd_;
  void *mapped_;
  bool retain_usage_;
};

// MemInfoTracker: tracks memory usage info via POSIX shared memory
class MemInfoTracker {
public:
  explicit MemInfoTracker(int64_t total_mem_size, int64_t group_id = 0,
                          const std::string &ipc_name = "")
      : ipc_name_(ipc_name), total_mem_size_(total_mem_size) {
    if (ipc_name_.empty()) {
      std::string base = obtain_default_ipc_name();
      // Non-zero group_id gets a "_g<id>" suffix so multiple pools
      // in one process don't share a segment.
      if (group_id != 0) {
        base += "_g" + std::to_string(group_id);
      }
      ipc_name_ = base;
    }
    init_kv_cache_limit(total_mem_size_);
    LOGGER(INFO,
           "MemInfoTracker initialized: ipc_name=%s, total_mem_size=%ld, "
           "group_id=%ld",
           ipc_name_.c_str(), total_mem_size_, group_id);
  }

  ~MemInfoTracker() { release_segment(); }

  MemInfoTracker(const MemInfoTracker &) = delete;
  MemInfoTracker &operator=(const MemInfoTracker &) = delete;

  // Update memory usage info in shared memory
  void update_memory_usage(int64_t used_size, int64_t prealloc_size) {
    RwLockedShm shm(ipc_name_, MemInfoStruct::SHM_SIZE, RwLockedShm::WLOCK);
    if (!shm.open()) {
      LOGGER(ERROR, "MemInfoTracker: failed to open shm for update: %s",
             ipc_name_.c_str());
      return;
    }
    MemInfoStruct info = shm.read_mem_info();
    info.used_size = used_size;
    info.prealloc_size = prealloc_size;
    shm.write_mem_info(info);
  }

  // Check if resize is needed, returns new mem_size (per layer), or -1 if not
  // needed
  int64_t check_and_get_resize_target(int64_t current_mem_size,
                                      int64_t num_layers,
                                      int64_t num_kv_buffers = 2) {
    RwLockedShm shm(ipc_name_, MemInfoStruct::SHM_SIZE, RwLockedShm::RLOCK);
    if (!shm.open()) {
      return -1;
    }
    MemInfoStruct info = shm.read_mem_info();
    int64_t new_mem_size = info.total_size / num_layers / num_kv_buffers;
    if (new_mem_size != current_mem_size) {
      return new_mem_size;
    }
    return -1;
  }

  const std::string &get_ipc_name() const { return ipc_name_; }

  // Stop local users before releasing. Never unlock explicitly: dup/fork
  // share the open file description, so F_UNLCK would drop a child's lease.
  // The kernel releases usage only after its last reference is closed. Stale
  // files are intentionally left for an explicit, locked kvctl delete.
  bool release_segment() {
    const int fd = retained_fd_.exchange(-1);
    if (fd >= 0)
      ::close(fd);
    return true;
  }

private:
  void init_kv_cache_limit(int64_t kv_cache_limit) {
    RwLockedShm shm(ipc_name_, MemInfoStruct::SHM_SIZE, RwLockedShm::WLOCK,
                    true);
    if (!shm.open()) {
      throw std::runtime_error("Cannot retain control segment " + ipc_name_ +
                               ": " + std::strerror(errno));
    }
    const int fd = shm.duplicate_fd();
    if (fd < 0)
      throw std::runtime_error("Cannot duplicate control segment " + ipc_name_);
    // Reset a stale same-name segment in place. Independently running engines
    // must use distinct names. A protocol upgrade requires all old participants
    // in this segment to stop first; an advisory lock cannot exclude old code.
    shm.write_mem_info(MemInfoStruct(kv_cache_limit, 0, 0));
    if (!control_segment::publish(fd)) {
      const int error = errno;
      ::close(fd);
      throw std::runtime_error("Cannot mark control segment " + ipc_name_ +
                               ": " + std::strerror(error));
    }
    retained_fd_ = fd;
  }

  // Get default IPC name (consistent with Python version logic)
  static std::string obtain_default_ipc_name() {
    // Prefer environment variable
    const char *env_name = std::getenv("KVCACHED_IPC_NAME");
    if (env_name && env_name[0] != '\0') {
      return std::string(env_name);
    }

    // Construct name using pgid
    pid_t pgid = getpgrp();
    char buf[256];
    snprintf(buf, sizeof(buf), "kvcached_engine_%d", static_cast<int>(pgid));
    return std::string(buf);
  }

  std::string ipc_name_;
  int64_t total_mem_size_;
  std::atomic<int> retained_fd_{-1};
};

} // namespace kvcached
