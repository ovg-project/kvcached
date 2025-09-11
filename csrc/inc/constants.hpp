// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kvcached {

using generic_ptr_t = void *;
using page_id_t = int64_t;
using offset_t = page_id_t;

// Page size is now configurable via kvcached initialization
extern size_t kPageSize;
static constexpr size_t kStartAddr = 0x1f0'000'000'000;

static constexpr page_id_t INV_PAGE_ID = -1;
static constexpr page_id_t ZERO_PAGE_ID = INV_PAGE_ID - 1;

static constexpr std::string_view k_prefix = "k_";
static constexpr std::string_view v_prefix = "v_";
static constexpr std::string_view kv_prefix = "kv_";

} // namespace kvcached
