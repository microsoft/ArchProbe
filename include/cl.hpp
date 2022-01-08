// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// OpenCL wrappings
// @PENGUINLIONG
#pragma once
#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION CL_TARGET_OPENCL_VERSION
#include <CL/opencl.hpp>

namespace archprobe {

class CLException : public std::exception {
  const char* msg;
public:
  CLException(cl_int code);

  const char* what() const noexcept override;
};
struct CLAssert {
  inline const CLAssert& operator<<(cl_int code) const {
    if (code != CL_SUCCESS) { throw CLException(code); }
    return *this;
  }
};
#define CL_ASSERT (::archprobe::CLAssert{})

struct DeviceStub {
  cl_platform_id platform_id;
  cl_device_id dev_id;
  std::string platform_exts;
  std::string dev_exts;
  std::string desc;
};

extern std::vector<DeviceStub> dev_stubs;

void initialize();
std::string desc_dev(uint32_t idx);
cl::Device select_dev(uint32_t idev);
cl::Context create_ctxt(const cl::Device& dev);
cl::CommandQueue create_cmd_queue(const cl::Context& ctxt);
cl::Program create_program(
  const cl::Device& dev,
  const cl::Context& ctxt,
  const char* src,
  const char* build_opts
);
inline cl::Program create_program(
  const cl::Device& dev,
  const cl::Context& ctxt,
  const std::string& src,
  const std::string& build_opts
) {
  return create_program(dev, ctxt, src.c_str(), build_opts.c_str());
}
cl::Kernel create_kernel(cl::Program program, const std::string& kernel_name);
double bench_kernel(
  const cl::CommandQueue& cmd_queue,
  const cl::Kernel& kernel,
  const cl::NDRange& local_size,
  const cl::NDRange& global_size,
  uint32_t niter
);

cl::Image2D create_img_2d(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  cl::ImageFormat img_fmt,
  uint32_t width,
  uint32_t height
);
inline cl::Image2D create_img_2d(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  cl_channel_order channel_order,
  cl_channel_type channel_type,
  uint32_t width,
  uint32_t height
) {
  cl::ImageFormat img_fmt(channel_order, channel_type);
  return create_img_2d(ctxt, mem_flags, img_fmt, width, height);
}
cl::Image1D create_img_1d(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  cl::ImageFormat img_fmt,
  uint32_t width
);
inline cl::Image1D create_img_1d(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  cl_channel_order channel_order,
  cl_channel_type channel_type,
  uint32_t width
) {
  cl::ImageFormat img_fmt(channel_order, channel_type);
  return create_img_1d(ctxt, mem_flags, img_fmt, width);
}
cl::Buffer create_buf(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  size_t size
);



struct MapImage {
  void* data;
  size_t width;
  size_t height;
  size_t depth;
  size_t row_pitch;
  size_t slice_pitch;

  operator void*() const {
    return data;
  }
};
struct MapBuffer {
  void* data;
  size_t size;

  operator void*() const {
    return data;
  }
};

MapImage map_img_2d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image2D& img
);
void unmap_img_2d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image2D& img,
  MapImage& mapped
);

MapImage map_img_1d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image1D& img
);
void unmap_img_1d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image1D& img,
  MapImage& mapped
);

MapBuffer map_buf(
  const cl::CommandQueue& cmd_queue,
  const cl::Buffer& buf,
  size_t offset,
  size_t size
);
MapBuffer map_buf(
  const cl::CommandQueue& cmd_queue,
  const cl::Buffer& buf
);
void unmap_buf(
  const cl::CommandQueue& cmd_queue,
  const cl::Buffer& buf,
  MapBuffer& mapped
);


} // namespace archprobe
