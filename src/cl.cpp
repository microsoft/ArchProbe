// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "cl.hpp"
#include "util.hpp"
#include "log.hpp"
#include "stats.hpp"

namespace archprobe {

CLException::CLException(cl_int code) {
  if (code <= -69) {
    msg = "invalid something";
    return;
  }
  switch (code) {
  case CL_INVALID_VALUE: msg = "invalid value"; break;
  case CL_INVALID_DEVICE_TYPE: msg = "invalid device type"; break;
  case CL_INVALID_PLATFORM: msg = "invalid platform"; break;
  case CL_INVALID_DEVICE: msg = "invalid device"; break;
  case CL_INVALID_CONTEXT: msg = "invalid context"; break;
  case CL_INVALID_QUEUE_PROPERTIES: msg = "invalid queue properties"; break;
  case CL_INVALID_COMMAND_QUEUE: msg = "invalid command queue"; break;
  case CL_INVALID_HOST_PTR: msg = "invalid host pointer"; break;
  case CL_INVALID_MEM_OBJECT: msg = "invalid memory object"; break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: msg = "invalid image format descriptor"; break;
  case CL_INVALID_IMAGE_SIZE: msg = "invalid image size"; break;
  case CL_INVALID_SAMPLER: msg = "invalid sampler"; break;
  case CL_INVALID_BINARY: msg = "invalid binary"; break;
  case CL_INVALID_BUILD_OPTIONS: msg = "invalid build options"; break;
  case CL_INVALID_PROGRAM: msg = "invalid program"; break;
  case CL_INVALID_PROGRAM_EXECUTABLE: msg = "invalid program executable"; break;
  case CL_INVALID_KERNEL_NAME: msg = "invalid kernel name"; break;
  case CL_INVALID_KERNEL_DEFINITION: msg = "invalid kernel definition"; break;
  case CL_INVALID_KERNEL: msg = "invalid kernel"; break;
  case CL_INVALID_ARG_INDEX: msg = "invalid arg index"; break;
  case CL_INVALID_ARG_VALUE: msg = "invalid arg value"; break;
  case CL_INVALID_ARG_SIZE: msg = "invalid arg size"; break;
  case CL_INVALID_KERNEL_ARGS: msg = "invalid kernel args"; break;
  case CL_INVALID_WORK_DIMENSION: msg = "invalid work dimension"; break;
  case CL_INVALID_WORK_GROUP_SIZE: msg = "invalid work group size"; break;
  case CL_INVALID_WORK_ITEM_SIZE: msg = "invalid work item_size"; break;
  case CL_INVALID_GLOBAL_OFFSET: msg = "invalid global offset"; break;
  case CL_INVALID_EVENT_WAIT_LIST: msg = "invalid event wait list"; break;
  case CL_INVALID_EVENT: msg = "invalid event"; break;
  case CL_INVALID_OPERATION: msg = "invalid operation"; break;
  case CL_INVALID_GL_OBJECT: msg = "invalid gl object"; break;
  case CL_INVALID_BUFFER_SIZE: msg = "invalid buffer size"; break;
  case CL_INVALID_MIP_LEVEL: msg = "invalid mip level"; break;
  case CL_INVALID_GLOBAL_WORK_SIZE: msg = "invalid global work size"; break;
  case CL_INVALID_PROPERTY: msg = "invalid property"; break;
  case CL_INVALID_IMAGE_DESCRIPTOR: msg = "invalid image descriptor"; break;
  case CL_INVALID_COMPILER_OPTIONS: msg = "invalid compiler options"; break;
  case CL_INVALID_LINKER_OPTIONS: msg = "invalid linker options"; break;
  case CL_INVALID_DEVICE_PARTITION_COUNT: msg = "invalid device partition count"; break;

  case CL_DEVICE_NOT_FOUND: msg = "device not found"; break;
  case CL_DEVICE_NOT_AVAILABLE: msg = "device not available"; break;
  case CL_COMPILER_NOT_AVAILABLE: msg = "compiler not available"; break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: msg = "memory object allocation failure"; break;
  case CL_OUT_OF_RESOURCES: msg = "out of resources"; break;
  case CL_OUT_OF_HOST_MEMORY: msg = "out of host memory"; break;
  case CL_PROFILING_INFO_NOT_AVAILABLE: msg = "profilng info not available"; break;
  case CL_MEM_COPY_OVERLAP: msg = "memory copy overlap"; break;
  case CL_IMAGE_FORMAT_MISMATCH: msg = "image format mismatch"; break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: msg = "image format not supported"; break;
  case CL_BUILD_PROGRAM_FAILURE: msg = "build program failure"; break;
  case CL_MISALIGNED_SUB_BUFFER_OFFSET: msg = "misaligned sub-buffer offset"; break;
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: msg = "execution status error for events in wait list"; break;
  case CL_COMPILE_PROGRAM_FAILURE: msg = "compile program failure"; break;
  case CL_LINKER_NOT_AVAILABLE: msg = "linker not available"; break;
  case CL_LINK_PROGRAM_FAILURE: msg = "link program failure"; break;
  case CL_DEVICE_PARTITION_FAILED: msg = "device partition failed"; break;
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: msg = "kernel argument information not available"; break;
  default: msg = "unknown opencl error"; break;
  }
}

const char* CLException::what() const noexcept { return msg; }



// Global variables.

std::vector<DeviceStub> dev_stubs;


std::vector<cl_platform_id> _enum_platform_ids() {
  cl_uint nplatform_id;
  CL_ASSERT << clGetPlatformIDs(0, nullptr, &nplatform_id);
  std::vector<cl_platform_id> platform_ids;
  platform_ids.resize(nplatform_id);
  CL_ASSERT << clGetPlatformIDs(nplatform_id, platform_ids.data(), nullptr);
  return platform_ids;
}
std::string _get_platform_info_str(
  cl_platform_id platform_id,
  cl_platform_info platform_info
) {
  size_t len = 0;
  std::string rv;
  CL_ASSERT << clGetPlatformInfo(platform_id, platform_info, 0, nullptr, &len);
  rv.reserve(len);
  rv.resize(len - 1);
  CL_ASSERT << clGetPlatformInfo(platform_id, platform_info,
    len, (char*)rv.data(), nullptr);
  return rv;
}
std::vector<cl_device_id> _enum_dev_ids(cl_platform_id platform_id) {
  cl_uint ndev_id;
  CL_ASSERT << clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,
    0, nullptr, &ndev_id);
  std::vector<cl_device_id> dev_ids;
  dev_ids.resize(ndev_id);
  CL_ASSERT << clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,
    ndev_id, dev_ids.data(), nullptr);
  return dev_ids;
}
std::string _get_dev_info_str(
  cl_device_id device_id,
  cl_device_info device_info
) {
  size_t len = 0;
  std::string rv;
  CL_ASSERT << clGetDeviceInfo(device_id, device_info, 0, nullptr, &len);
  rv.reserve(len);
  rv.resize(len - 1);
  CL_ASSERT << clGetDeviceInfo(device_id, device_info,
    len, (char*)rv.data(), nullptr);
  return rv;
}
void initialize() {
  auto platform_ids = _enum_platform_ids();
  for (auto platform_id : platform_ids) {
    auto platform_name =
      _get_platform_info_str(platform_id, CL_PLATFORM_NAME);
    auto platform_prof =
      _get_platform_info_str(platform_id, CL_PLATFORM_PROFILE);
    auto platform_exts =
      _get_platform_info_str(platform_id, CL_PLATFORM_EXTENSIONS);

    auto platform_desc =
      util::format(platform_name, " (", platform_prof, ") - ");
    
    auto dev_ids = _enum_dev_ids(platform_id);
    for (auto dev_id : dev_ids) {
      auto dev_name = _get_dev_info_str(dev_id, CL_DEVICE_NAME);
      auto dev_ver = _get_dev_info_str(dev_id, CL_DEVICE_VERSION);
      auto dev_exts = _get_dev_info_str(dev_id, CL_DEVICE_EXTENSIONS);
      cl_device_type dev_ty;
      CL_ASSERT << clGetDeviceInfo(dev_id, CL_DEVICE_TYPE,
        sizeof(dev_ty), &dev_ty, nullptr);
      const char* dev_ty_lit;
      switch (dev_ty) {
      case CL_DEVICE_TYPE_CPU: dev_ty_lit = "CPU"; break;
      case CL_DEVICE_TYPE_GPU: dev_ty_lit = "GPU"; break;
      case CL_DEVICE_TYPE_ACCELERATOR: dev_ty_lit = "Accelerator"; break;
      default: dev_ty_lit = "Unknown"; break;
      }

      auto desc = platform_desc +
        util::format(dev_name, " (", dev_ty, ", ", dev_ver, ")");

      DeviceStub stub { platform_id, dev_id, platform_exts, dev_exts, desc };
      dev_stubs.emplace_back(std::move(stub));
    }
  }
  archprobe::log::info("initialized opencl environment");
}
std::string desc_dev(uint32_t idx) {
  return idx < dev_stubs.size() ? dev_stubs[idx].desc : std::string {};
}

cl::Device select_dev(uint32_t idev) {
  const auto& dev_stub = archprobe::dev_stubs[idev];
  archprobe::log::info("selected device #", idev, ": ", dev_stub.desc);
  cl::Device dev(dev_stub.dev_id);

  return dev;
}

cl::Context create_ctxt(const cl::Device& dev) {

  // Create context.
  cl_context_properties ctxt_props[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)dev.getInfo<CL_DEVICE_PLATFORM>(),
    0,
  };

  cl_int err;
  cl::Context ctxt = cl::Context(dev, ctxt_props, nullptr, nullptr, &err);
  CL_ASSERT << err;

  return ctxt;
}

cl::CommandQueue create_cmd_queue(const cl::Context& ctxt) {
  cl_int err;
  cl::CommandQueue cmd_queue(ctxt, CL_QUEUE_PROFILING_ENABLE, &err);
  CL_ASSERT << err;

  return cmd_queue;
}

cl::Program create_program(
  const cl::Device& dev,
  const cl::Context& ctxt,
  const char* src,
  const char* build_opts
) {
  cl_int err;
  cl::Program::Sources sources;
  sources.push_back(src);
  cl::Program program(ctxt, sources, &err);
  CL_ASSERT << err;

  err = program.build({dev}, build_opts);
  if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev) == CL_BUILD_ERROR) {
    std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
    archprobe::log::error(build_log);
  }
  CL_ASSERT << err;

  return program;
}

cl::Kernel create_kernel(cl::Program program, const std::string& kernel_name) {
  cl_int err;
  cl::Kernel kernel(program, kernel_name.c_str(), &err);
  CL_ASSERT << err;
  return kernel;
}


double bench_kernel(
  const cl::CommandQueue& cmd_queue,
  const cl::Kernel& kernel,
  const cl::NDRange& local_size,
  const cl::NDRange& global_size,
  uint32_t niter
) {
  std::vector<cl::Event> events;

  auto run_kernel = [&]() {
    cl::Event event;
    CL_ASSERT << cmd_queue.enqueueNDRangeKernel(kernel, cl::NDRange(0, 0, 0),
      global_size, local_size, nullptr, &event);
    events.push_back(event);
  };

  run_kernel();
  run_kernel();
  cmd_queue.finish();
  events.clear();

  for (auto i = 0; i < niter; ++i) {
    run_kernel();
  }
  cmd_queue.finish();

  archprobe::stats::MedianStats<double> time_avg;
  for (const auto& event : events) {
    uint64_t start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    uint64_t end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    time_avg.push(end - start);
  }
  double time = (double)time_avg / 1000;
  events.clear();
  return time;
}

cl::Image2D create_img_2d(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  cl::ImageFormat img_fmt,
  uint32_t width,
  uint32_t height
) {
  cl_int err;
  cl::Image2D img(ctxt, mem_flags, img_fmt, width, height, 0, nullptr, &err);
  CL_ASSERT << err;
  return img;
}
cl::Image1D create_img_1d(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  cl::ImageFormat img_fmt,
  uint32_t width
) {
  cl_int err;
  cl::Image1D img(ctxt, mem_flags, img_fmt, width, nullptr, &err);
  CL_ASSERT << err;
  return img;
}
cl::Buffer create_buf(
  const cl::Context& ctxt,
  cl_mem_flags mem_flags,
  size_t size
) {
  cl_int err;
  cl::Buffer buf(ctxt, mem_flags, size, 0, &err);
  CL_ASSERT << err;
  return buf;
}

MapImage map_img_2d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image2D& img
) {
  cl_int err;

  const size_t width = img.getImageInfo<CL_IMAGE_WIDTH>();
  const size_t height = img.getImageInfo<CL_IMAGE_HEIGHT>();

  size_t row_pitch;
  size_t slice_pitch;
  cl::array<size_t, 3> origin {};
  cl::array<size_t, 3> region { width, height, 1 };

  float* data = (float*)cmd_queue.enqueueMapImage(img, true,
    CL_MAP_READ | CL_MAP_WRITE, origin, region, &row_pitch, &slice_pitch,
    nullptr, nullptr, &err);
  CL_ASSERT << err;
  return MapImage { data, width, height, 1, row_pitch, slice_pitch };
}
void unmap_img_2d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image2D& img,
  MapImage& mapped
) {
  CL_ASSERT << cmd_queue.enqueueUnmapMemObject(img, mapped);
  mapped = {};
}

MapImage map_img_1d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image1D& img
) {
  cl_int err;

  const size_t width = img.getImageInfo<CL_IMAGE_WIDTH>();

  size_t row_pitch;
  size_t slice_pitch;
  cl::array<size_t, 3> origin {};
  cl::array<size_t, 3> region { width, 1, 1 };

  float* data = (float*)cmd_queue.enqueueMapImage(img, true,
    CL_MAP_READ | CL_MAP_WRITE, origin, region, &row_pitch, &slice_pitch,
    nullptr, nullptr, &err);
  CL_ASSERT << err;
  return MapImage { data, width, 1, 1, row_pitch, slice_pitch };
}
void unmap_img_1d(
  const cl::CommandQueue& cmd_queue,
  const cl::Image1D& img,
  MapImage& mapped
) {
  CL_ASSERT << cmd_queue.enqueueUnmapMemObject(img, mapped);
  mapped = {};
}

MapBuffer map_buf(
  const cl::CommandQueue& cmd_queue,
  const cl::Buffer& buf,
  size_t offset,
  size_t size
) {
  cl_int err;
  auto rv = (float*)cmd_queue.enqueueMapBuffer(buf, true,
    CL_MAP_READ | CL_MAP_WRITE, offset, size, nullptr, nullptr, &err);
  CL_ASSERT << err;
  return MapBuffer { rv, size };
}
MapBuffer map_buf(
  const cl::CommandQueue& cmd_queue,
  const cl::Buffer& buf
) {
  const size_t size = buf.getInfo<CL_MEM_SIZE>();
  return map_buf(cmd_queue, buf, 0, size);
}
void unmap_buf(
  const cl::CommandQueue& cmd_queue,
  const cl::Buffer& buf,
  MapBuffer& mapped
) {
  CL_ASSERT << cmd_queue.enqueueUnmapMemObject(buf, mapped);
  mapped = {};
}

} // namespace archprobe
