/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors
  Copyright (C) 2023 Intel Corporation

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#pragma once

#include <sycl/sycl.hpp>
#include "dpct/dpct.hpp"
#include "dpct/blas_utils.hpp"

#include "utils/exception.h"

#if defined(__HIP_PLATFORM_AMD__) && (defined(__GFX9__) || defined(__GFX8__))
#define SYCL_SUB_GROUP_SIZE 64
#else
#define SYCL_SUB_GROUP_SIZE 32
#endif

namespace lczero {
namespace sycldnn_backend {

static constexpr int kNumOutputPolicy = 1858;

// max supported filter count for fast path
// TODO: extend it to cover bigger networks!
// (We are limited by no of registers per thread)
static constexpr int kMaxResBlockFusingChannels = 384;  // limit on num_filters
static constexpr int kMaxResBlockFusingSeKFp16Ampere =
    512;  // (use a different kernel with reduced register pressure)
static constexpr int kMaxResBlockFusingSeK =
    128;  // limit on (num_filters / se_ratio)
static constexpr int kMaxResBlockFusingSeFp16AmpereSmem =
    72 * kMaxResBlockFusingSeKFp16Ampere *
    sizeof(sycl::half);  // shared memory used by the special
                         // kernel

#ifdef USE_CUBLAS
void CublasError(int status, const char* file, const int& line);

#define ReportCUBLASErrors(status) CublasError(status, __FILE__, __LINE__)
#endif

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Device capability storage
struct DeviceCapabilities {
private:
  static size_t max_workgroup_size_;
  static bool initialized_;

public:
  static void Initialize(const sycl::queue& queue) {
    if (!initialized_) {
      max_workgroup_size_ = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
      initialized_ = true;
    }
  }
  
  static size_t GetMaxWorkgroupSize() {
    if (!initialized_) {
      throw Exception("DeviceCapabilities not initialized");
    }
    return max_workgroup_size_;
  }
  
  static int GetOptimalBlockSize() {
    size_t max_size = GetMaxWorkgroupSize();
    // Use the largest power of 2 that doesn't exceed max_workgroup_size
    // Common values are 64, 128, 256, 512, 1024, but cap at device limit
    int optimal = 256; // Default fallback
    if (max_size >= 1024) optimal = 1024;
    else if (max_size >= 512) optimal = 512;
    else if (max_size >= 256) optimal = 256;
    else if (max_size >= 128) optimal = 128;
    else if (max_size >= 64) optimal = 64;
    else optimal = static_cast<int>(max_size);
    
    return std::min(optimal, static_cast<int>(max_size));
  }
};

// Declaration of static members
inline size_t DeviceCapabilities::max_workgroup_size_ = 0;
inline bool DeviceCapabilities::initialized_ = false;

}  // namespace sycldnn_backend
}  // namespace lczero
