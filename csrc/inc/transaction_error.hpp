// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>

namespace kvcached {

// Only a newly allocated, unpublished page may be quarantined and bypassed.
class MapQuarantinedError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// The caller must stop using the pool; retrying is not a capacity miss.
class StateConsistencyError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class QuarantinedResizeError : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

} // namespace kvcached
