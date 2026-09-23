// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

  RwLockedShm(const std::string &file_path, size_t size, LockType lock_type)
      : file_path_(get_ipc_path(file_path)), size_(size), lock_type_(lock_type),
        fd_(-1), mapped_(nullptr) {}

  ~RwLockedShm() { close(); }

  // Open and lock the shared memory file, returns whether successful
  bool open() {
    // Try to open the file
    fd_ = ::open(file_path_.c_str(), O_RDWR);
    if (fd_ < 0) {
      if (lock_type_ != WLOCK) {
        return false;
      }
      // Create file in write-lock mode
      fd_ = ::open(file_path_.c_str(), O_RDWR | O_CREAT, 0666);
      if (fd_ < 0) {
        return false;
      }
      if (ftruncate(fd_, size_) < 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
      }
    }

    // Ensure file size is sufficient
    struct stat st;
    if (fstat(fd_, &st) == 0 && static_cast<size_t>(st.st_size) < size_) {
      if (lock_type_ == WLOCK) {
        (void)ftruncate(fd_, size_);
      }
    }

    // Acquire file lock
    if (flock(fd_, lock_type_) < 0) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }

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

  // All users of this tracker must be stopped first. Multiple local pools can
  // share an explicit IPC name; only the last stopped pool removes that inode.
  bool release_segment() {
    std::lock_guard<std::mutex> guard(segment_registry_mutex());
    if (!released_) {
      if (cleanup_) {
        --cleanup_->users;
      } else if (identity_capture_failed_) {
        auto &unknown = uncaptured_users();
        if (--unknown.at(cleanup_path_) == 0)
          unknown.erase(cleanup_path_);
      }
      released_ = true;
    }
    if (!cleanup_)
      return !identity_capture_failed_;
    if (uncaptured_users().count(cleanup_->path) != 0) {
      LOGGER(WARNING,
             "MemInfoTracker: keeping %s while a pool with an "
             "unconfirmed identity is still active",
             cleanup_->path.c_str());
      return false;
    }
    return cleanup_->users != 0 || cleanup_->unlink_original();
  }

private:
  static std::mutex &segment_registry_mutex() {
    static std::mutex mutex;
    return mutex;
  }

  static std::map<std::string, size_t> &uncaptured_users() {
    static std::map<std::string, size_t> users;
    return users;
  }

  struct SegmentCleanup {
    explicit SegmentCleanup(std::string path, int fd)
        : path(std::move(path)), fd(fd) {}
    ~SegmentCleanup() {
      if (fd >= 0)
        ::close(fd);
    }

    // Called under the lifecycle mutex; failures retain the fd for retries.
    bool unlink_original() {
      if (fd < 0)
        return true;
      struct stat original;
      struct stat current;
      if (::fstat(fd, &original) < 0) {
        LOGGER(WARNING,
               "MemInfoTracker: cannot read retained identity for %s: %s",
               path.c_str(), std::strerror(errno));
        return false;
      }
      if (::stat(path.c_str(), &current) < 0) {
        if (errno != ENOENT) {
          LOGGER(WARNING,
                 "MemInfoTracker: cannot verify shm identity for %s: %s",
                 path.c_str(), std::strerror(errno));
          return false;
        }
      } else if (original.st_dev == current.st_dev &&
                 original.st_ino == current.st_ino &&
                 ::unlink(path.c_str()) < 0 && errno != ENOENT) {
        LOGGER(WARNING, "MemInfoTracker: failed to unlink shm %s: %s",
               path.c_str(), std::strerror(errno));
        return false;
      }
      ::close(fd);
      fd = -1;
      return true;
    }

    std::string path;
    int fd;
    size_t users = 0;
  };

  static std::shared_ptr<SegmentCleanup> retain_segment(RwLockedShm &shm) {
    static std::map<std::pair<dev_t, ino_t>, std::weak_ptr<SegmentCleanup>>
        registry;
    const int fd = shm.duplicate_fd();
    struct stat identity;
    if (fd < 0 || ::fstat(fd, &identity) < 0) {
      const int error = errno;
      if (fd >= 0)
        ::close(fd);
      LOGGER(WARNING, "MemInfoTracker: cannot retain identity for %s: %s",
             shm.path().c_str(), std::strerror(error));
      return nullptr;
    }
    for (auto it = registry.begin(); it != registry.end();) {
      if (it->second.expired())
        it = registry.erase(it);
      else
        ++it;
    }
    auto &entry = registry[{identity.st_dev, identity.st_ino}];
    auto cleanup = entry.lock();
    if (!cleanup || cleanup->fd < 0) {
      cleanup = std::make_shared<SegmentCleanup>(shm.path(), fd);
      entry = cleanup;
    } else {
      ::close(fd);
    }
    ++cleanup->users;
    return cleanup;
  }

  // Initialize kv cache limit in shared memory
  void init_kv_cache_limit(int64_t kv_cache_limit) {
    // Serialize local creation with last-owner release. No allocation or
    // memory-usage update takes this lifecycle-only lock.
    std::lock_guard<std::mutex> guard(segment_registry_mutex());
    RwLockedShm shm(ipc_name_, MemInfoStruct::SHM_SIZE, RwLockedShm::WLOCK);
    if (!shm.open()) {
      LOGGER(ERROR, "MemInfoTracker: failed to create shm: %s",
             ipc_name_.c_str());
      return;
    }
    // Pin the opened inode, not a second lookup of a replaceable pathname.
    cleanup_ = retain_segment(shm);
    identity_capture_failed_ = !cleanup_;
    if (identity_capture_failed_) {
      cleanup_path_ = shm.path();
      ++uncaptured_users()[cleanup_path_];
    }
    MemInfoStruct info(kv_cache_limit, 0, 0);
    shm.write_mem_info(info);
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
  std::shared_ptr<SegmentCleanup> cleanup_;
  std::string cleanup_path_;
  bool identity_capture_failed_ = false;
  bool released_ = false;
};

} // namespace kvcached
