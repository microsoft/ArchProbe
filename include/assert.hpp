// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/d291c3d1ce3795fe4b305e5efd76b4f586d23e3b/include/assert.hpp;
// MIT-licensed by Rendong Liang.

// Assertion.
// @PENGUINLIONG
#pragma once
#include "util.hpp"
#undef assert

namespace archprobe {

class AssertionFailedException : public std::exception {
  std::string msg;
public:
  AssertionFailedException(const std::string& msg);

  const char* what() const noexcept override;
};

template<typename ... TArgs>
inline void assert(bool pred, const TArgs& ... args) {
  if (!pred) {
    throw AssertionFailedException(util::format(args ...));
  }
}
template<typename ... TArgs>
inline void panic(const TArgs& ... args) {
  assert<TArgs ...>(false, args ...);
}
template<typename ... TArgs>
inline void unreachable(const TArgs& ... args) {
  assert<const char*, TArgs ...>(false, "reached unreachable code: ", args ...);
}

} // namespace archprobe
