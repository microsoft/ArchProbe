// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Numeric data table.
// @PENGUINLIONG
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include "assert.hpp"

namespace archprobe {
namespace table {

struct Table {
  std::vector<std::string> headers;
  std::vector<std::vector<double>> rows;

  template<typename ... THeaders>
  Table(THeaders&& ... headers) :
    Table(std::vector<std::string> { std::string(headers) ... }) {}
  Table(std::vector<std::string>&& headers);
  Table(
    std::vector<std::string>&& headers,
    std::vector<std::vector<double>>&& rows);

  template<typename ... TArgs>
  void push(TArgs&& ... values) {
    std::vector<double> row { (double)values ... };
    archprobe::assert(row.size() == headers.size(),
      "row length mismatches header length");
    rows.emplace_back(std::move(row));
  }

  std::string to_csv(uint32_t nsig_digit = 6) const;
  static Table from_csv(std::string csv);
};

} // namespace table
} // namespace archprobe
