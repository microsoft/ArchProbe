// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <signal.h>
#include "env.hpp"

using namespace archprobe;
using namespace archprobe::stats;

namespace {

void log_cb(log::LogLevel lv, const std::string& msg) {
  using log::LogLevel;
  switch (lv) {
  case LogLevel::L_LOG_LEVEL_DEBUG:
    printf("[\x1b[90mDEBUG\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_INFO:
    printf("[\x1B[32mINFO\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_WARNING:
    printf("[\x1B[33mWARN\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_ERROR:
    printf("[\x1B[31mERROR\x1B[0m] %s\n", msg.c_str());
    break;
  }
  std::fflush(stdout);
}

} // namespace

namespace archprobe {

cl_channel_order channel_order_by_ncomp(uint32_t ncomp) {
  switch (ncomp) {
  case 1: return CL_R;
  case 2: return CL_RG;
  case 4: return CL_RGBA;
  default:
    panic("image component count must be 1, 2 or 4");
  }
  return CL_RGBA;
}

std::string vec_name_by_ncomp(const char* scalar_name, uint32_t ncomp) {
  return scalar_name + (ncomp == 1 ? "" : std::to_string(ncomp));
}
std::string vec_name_by_ncomp_log2(
  const char* scalar_name,
  uint32_t ncomp_log2
) {
  return vec_name_by_ncomp(scalar_name, 1 << ncomp_log2);
}



template<typename T>
bool is_pow2(T x) {
  return (x & (-x)) == x;
}
template<typename T>
T next_pow2(T x) {
  T y = 1;
  for (; y < x; y <<= 1) {}
  return y;
}

// Power of integer.
template<typename T>
T powi(T x, T p) {
  if (p == 0) { return 1; }
  if (p == 1) { return x; }
  return x * powi(x, p - 1);
}
// Log2 of integer. 0 is treated as two-to-the-zero.
template<typename T>
T log2i(T x) {
  T counter = 0;
  while (x >= 2) {
    x >>= 1;
    ++counter;
  }
  return counter;
}


using Aspect = std::function<void(Environment&)>;



class ArchProbe {
  Environment env_;

public:
  ArchProbe(uint32_t idev) : env_(idev) {}

  ArchProbe& with_aspect(const Aspect& aspect) {
    aspect(env_);
    return *this;
  }

  void clear_aspect_report(const std::string& aspect) {
    env_.clear_aspect_report(aspect);
  }
};

template<uint32_t NTap>
struct DtJumpFinder {
private:
  NTapAvgStats<double, NTap> time_avg_;
  AvgStats<double> dtime_avg_;
  double compensation_;
  double threshold_;

public:
  // Compensation is a tiny additive to give on delta time so that the algorithm
  // works smoothly when a sequence of identical timing is ingested, which is
  // pretty common in our tests. Threshold is simply how many times the new
  // delta has to be to be recognized as a deviation.
  DtJumpFinder(double compensation = 0.01, double threshold = 10) :
    time_avg_(), dtime_avg_(), compensation_(compensation),
    threshold_(threshold) {}

  // Returns true if the delta time regarding to the last data point seems
  // normal; returns false if it seems the new data point is too much away from
  // the historical records.
  bool push(double time) {
    if (time_avg_.has_value()) {
      double dtime = std::abs(time - time_avg_) + (compensation_ * time_avg_);
      if (dtime_avg_.has_value()) {
        double ddtime = std::abs(dtime - dtime_avg_);
        if (ddtime > threshold_ * dtime_avg_) {
          return true;
        }
      }
      dtime_avg_.push(dtime);
    }
    time_avg_.push(time);
    return false;
  }

  double dtime_avg() const { return dtime_avg_; }
  double compensate_time() const { return compensation_ * time_avg_; }
};


namespace aspects {

  // This aspect tests the number of registers owned by each thread. On some
  // architectures, like Arm Mali, the workgroup size can be larger if each
  // thread is using a smaller number of registers, so the report will be the
  // maximal number of threads at which number of registers being available.
  //
  // In the experiment, an array of 32b words, here floats, is allocated, the
  // registers will hold a number of data, and finally be wriiten
  // back to memory to prevent optimization. When the number of register used
  // exceeds the physical register file size, i.e. register spill, some or all
  // the registers will be fallbacked to memory, and kernel latency is
  // significantly increased.
  //
  // TODO: (penguinliong) It can get negative delta time many times. Very likely
  // it can be the kernel exited too early when too many registers are
  // allocated. Need a way to know the upper limit in advance, or it's just a
  // driver bug.
  void reg_count(Environment& env) {
    if (env.report_started_lazy("RegCount")) { return; }
    env.check_dep("Device");
    env.init_table("nthread", "ngrp", "nreg", "niter", "t (us)");
    bool done = true;

    const uint32_t NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    // This number should be big enough to contain the number of registers
    // reserved for control flow, but not too small to ignore register count
    // boundaries in first three iterations.
    const double COMPENSATE = env.cfg_num("Compensate", 0.01);
    const double THRESHOLD = env.cfg_num("Threshold", 10);

    const uint32_t NREG_MIN = env.cfg_num("NRegMin", 1);
    const uint32_t NREG_MAX = env.cfg_num("NRegMax", 512);
    const uint32_t NREG_STEP = env.cfg_num("NRegStep", 1);

    const uint32_t NGRP_MIN = env.cfg_num("NGrpMin", 1);
    const uint32_t NGRP_MAX = env.cfg_num("NGrpMax", 64);
    const uint32_t NGRP_STEP = env.cfg_num("NGrpStep", 1);

    uint32_t NITER;

    cl::Buffer out_buf = env.create_buf(0, sizeof(float));
    auto bench = [&](uint32_t nthread, uint32_t ngrp, uint32_t nreg) {
      std::string reg_declr = "";
      std::string reg_comp = "";
      std::string reg_reduce = "";
      for (int i = 0; i < nreg; ++i) {
        reg_declr += util::format("float reg_data", i, " = "
          "(float)niter + ", i, ";\n");
      }
      for (int i = 0; i < nreg; ++i) {
        int last_i = i == 0 ? nreg - 1 : i - 1;
        reg_comp += util::format("reg_data", i, " *= reg_data", last_i,
          ";\n");
      }
      for (int i = 0; i < nreg; ++i) {
        reg_reduce += util::format("out_buf[", i, " * i] = reg_data",
          i, ";\n");
      }

      auto src = util::format(R"(
        __kernel void reg_count(
          __global float* out_buf,
          __private const int niter
        ) {
          )", reg_declr, R"(
          int i = 0;
          for (; i < niter; ++i) {
          )", reg_comp, R"(
          }
          i = i >> 31;
          )", reg_reduce, R"(
        }
      )");
      //log::debug(src);
      cl::Program program = env.create_program(src, "");
      cl::Kernel kernel = env.create_kernel(program, "reg_count");
      kernel.setArg(0, out_buf);
      kernel.setArg(1, (int)NITER);
      // Make sure all SMs are used. It's good for accuracy.
      cl::NDRange global(nthread, ngrp, 1);
      cl::NDRange local(nthread, 1, 1);
      double time = env.bench_kernel(kernel, local, global, 10);
      env.table().push(nthread, ngrp, nreg, NITER, time);
      return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() {
      return bench(1, 1, NREG_MIN);
    });

    // Step 1. Probe for the most number of registers we can use by activating
    // different numbers of threads to try to use up all the register resources.
    // In case of register spill the kernel latency would surge.
    //
    // The group size is kept 1 to minimize the impact of scheduling.
    uint32_t nreg_max;
    if (!env.try_get_report("RegCount", nreg_max) && done) {
      log::info("testing register availability when only 1 thread is "
        "dispatched");

      DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
      uint32_t nreg = NREG_MIN;
      for (; nreg <= NREG_MAX; nreg += NREG_STEP) {
        double time = bench(1, 1, nreg);
        log::debug("testing nreg=", nreg, ", time=", time);

        if (dj.push(time)) {
          nreg -= NREG_STEP;
          log::info(nreg, " registers are available at most");
          nreg_max = nreg;
          break;
        }
      }
      if (nreg >= NREG_MAX) {
        log::warn("unable to conclude a maximal register count");
        done = false;
        nreg_max = NREG_STEP;
      } else {
        env.report_value("RegCount", nreg_max);
      }
    }

    // Step 2: Knowing the maximal number of registers available to a single
    // thread, we wanna check if the allocation is pooled. A pool of register is
    // shared by all threads concurrently running, which means the more register
    // we use per thread, the less number of concurrent threads we can have in a
    // warp. Then we would have to consider the degradation of parallelism due
    // to register shortage.
    //
    // In this implementation we measure at which number of workgroups full-
    // occupation and half-occupation of registers are having a jump in latency.
    // If the registers are pooled, the full-occupation case should only support
    // half the number of threads of the half-occupation case. Otherwise, the
    // full-occupation case jump at the same point as half-occupation, the
    // registers should be dedicated to each physical thread.
    auto find_ngrp_by_nreg = [&](uint32_t nreg) {
      DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
      for (auto ngrp = NGRP_MIN; ngrp <= NGRP_MAX; ngrp += NGRP_STEP) {
        auto time = bench(1, ngrp, nreg);
        log::debug("testing occupation (nreg=", nreg, "); ngrp=", ngrp,
          ", time=", time, " us");

        if (dj.push(time)) {
          ngrp -= NGRP_STEP;
          log::info("using ", nreg, " registers can have ", ngrp,
            " concurrent single-thread workgroups");
          return ngrp;
        }
      }
      log::warn("unable to conclude a maximum number of concurrent "
        "single-thread workgroups when ", nreg, " registers are occupied");
      done = false;
      return (uint32_t)1;
    };

    uint32_t ngrp_full, ngrp_half;
    if (!env.try_get_report("FullRegConcurWorkgroupCount", ngrp_full) && done) {
      ngrp_full = find_ngrp_by_nreg(nreg_max);
      env.report_value("FullRegConcurWorkgroupCount", ngrp_full);
    }
    if (!env.try_get_report("HalfRegConcurWorkgroupCount", ngrp_half) && done) {
      ngrp_half = find_ngrp_by_nreg(nreg_max / 2);
      env.report_value("HalfRegConcurWorkgroupCount", ngrp_half);
    }

    std::string reg_ty;
    if (!env.try_get_report("RegType", reg_ty) && done) {
      if (ngrp_full * 1.5 < ngrp_half) {
        log::info("all physical threads in an sm share ", nreg_max,
          " registers");
        env.report_value("RegType", "Pooled");
      } else {
        log::info("each physical thread has ", nreg_max, " registers");
        env.report_value("RegType", "Dedicated");
      }
    }

    env.report_ready(done);
  }

  // This aspect tests the cacheline size of the buffer data pathway. A
  // cacheline is the basic unit of storage in a cache hierarchy, and the
  // cacheline size of the top-level cache system directly determines the
  // optimal alignment of memory access. No matter how much data the kernel
  // would fetch from lower hierarchy, the entire cacheline will be fetched in.
  // If we are not reading all the data in a cacheline, a portion of memory
  // bandwidth will be wasted, which is an undesired outcome.
  //
  // In this experiment all logically concurrent threads read from the memory
  // hierarchy with a varying stride. The stride is initially small and many
  // accesses will hit the cachelines densely; but as the stride increases there
  // will be only a single float taken from each cacheline, which leads to
  // serious cache flush and increased latency.
  void buf_cacheline_size(Environment& env) {
    if (env.report_started_lazy("BufferCachelineSize")) { return; }
    env.init_table("nthread", "stride (byte)", "pitch (byte)", "niter",
      "t (us)");
    bool done = true;

    const uint32_t NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    const uint32_t BUF_CACHE_SIZE =
      env.must_get_aspect_report<uint32_t>("Device", "CacheSize");

    const double COMPENSATE = env.cfg_num("Compensate", 0.01);
    const double THRESHOLD = env.cfg_num("Threshold", 10);

    const uint32_t PITCH = BUF_CACHE_SIZE * 2 / NTHREAD_LOGIC;
    const uint32_t BUF_SIZE = PITCH * NTHREAD_LOGIC;
    const uint32_t MAX_STRIDE = PITCH / 2;

    uint32_t NITER;

    const char* src = R"(
      __kernel void buf_cacheline_size(
        __global const float* src,
        __global float* dst,
        __private const int niter,
        __private const int stride,
        __private const int pitch
      ) {
        float c = 0;
        for (int i = 0; i < niter; ++i) {
          const int zero = i >> 31;
          c += src[zero + stride * 0 + pitch * get_global_id(0)];
          c += src[zero + stride * 1 + pitch * get_global_id(0)];
        }
        dst[0] = c;
      }
    )";
    cl::Program program = env.create_program(src, "");
    cl::Kernel kernel = env.create_kernel(program, "buf_cacheline_size");

    cl::Buffer in_buf = env.create_buf(0, BUF_SIZE);
    cl::Buffer out_buf = env.create_buf(0, sizeof(float));

    auto bench = [&](int stride) {
      cl::NDRange global(NTHREAD_LOGIC, 1, 1);
      cl::NDRange local(NTHREAD_LOGIC, 1, 1);
      kernel.setArg(0, in_buf);
      kernel.setArg(1, out_buf);
      kernel.setArg(2, (int)(NITER));
      kernel.setArg(3, (int)(stride / sizeof(float)));
      kernel.setArg(4, (int)(PITCH / sizeof(float)));
      auto time = env.bench_kernel(kernel, local, global, 10);
      env.table().push(NTHREAD_LOGIC, stride, PITCH, NITER, time);
      return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(sizeof(float)); });

    uint32_t cacheline_size;
    if (!env.try_get_report("BufCachelineSize", cacheline_size) && done) {
      log::info("testing buffer cacheline size");

      DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
      uint32_t stride = sizeof(float);
      for (; stride <= MAX_STRIDE; stride += sizeof(float)) {
        double time = bench(stride);
        log::debug("testing stride=", stride, ", time=", time);

        if (dj.push(time)) {
          cacheline_size = stride;
          log::info("top level buffer cacheline size is ", cacheline_size, "B");
          break;
        }
      }
      if (stride >= MAX_STRIDE) {
        log::warn("unable to conclude a top level buffer cacheline size");
        done = false;
      } else {
        env.report_value("BufTopLevelCachelineSize", cacheline_size);
      }
    }

    env.report_ready(done);
  }

  // This aspect tests the top-level cache system for textures, a.k.a. the
  // texture memory. In computer graphics, texture is devised to provide 2D
  // representations of surface color for 3D models, usually a triangle mesh.
  // To improve render quality, the textures are usually filtered and
  // interpolated to smooth out color steps in the original image, which
  // requires multiple access to a 2D area nearby the point of interpolation.
  // Textures are therefore drastically different from buffers, they need to
  // provide rapid access to data points in a 2D patch; and each data point has
  // four elements, as it's originally been designed for, corresponding to the
  // components in the RGB color space with an extra alpha channel. Also, unlike
  // a buffer which refers to a contiguous range of memory, a texture is an
  // opaque object, defined by GPU vendors and the users usually have no
  // knowledge about its actual data layout, so it is possible that the memory
  // *does not* layout linearly. These characteristics prevent us from designing
  // the addressing and vectorization of data. The only general optimization we
  // can do is to align the amount of data accessed at a time to the relatively
  // small top-level cache system, i.e., the L1 texture cache.
  //
  // In the experiment, we assume L1 cacheline size is small enough that
  // several threads reading float4s can exceed it. In both direction, along the
  // width and the height, each logically concurrent thread in an SM reads a
  // float4. Such memory access should be satisfied by a single cache fetch, but
  // if the cache is not large enough to contain all requested data, multiple
  // fetches will significantly increase access latency. A large iteration
  // number is used to magnify the latency.
  void img_cacheline_size(Environment& env) {
    if (env.report_started_lazy("ImageCachelineSize")) { return; }
    env.init_table("nthread", "dim (x/y)", "niter", "t (us)");
    bool done = true;

    const uint32_t NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    const uint32_t MAX_IMG_WIDTH =
      env.must_get_aspect_report<uint32_t>("Device", "MaxImageWidth");
    const uint32_t MAX_IMG_HEIGHT =
      env.must_get_aspect_report<uint32_t>("Device", "MaxImageHeight");
    const uint32_t PX_SIZE = 4 * sizeof(float);

    const double COMPENSATE = env.cfg_num("Compensate", 0.01);
    const double THRESHOLD = env.cfg_num("Threshold", 10);

    uint32_t NITER;

    const char* src = R"(
      __constant sampler_t SAMPLER =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      __kernel void img_cacheline_size_x(
        __read_only image2d_t src,
        __global float* dst,
        __private const int niter
      ) {
        float4 c = 0;
        for (int j = 0; j < niter; ++j) {
          int zero = j >> 31;
          c += read_imagef(src, SAMPLER, (int2)(get_global_id(0), zero));
        }
        dst[0] = c.x * c.y * c.z * c.w;
      }
      __kernel void img_cacheline_size_y(
        __read_only image2d_t src,
        __global float* dst,
        __private const int niter
      ) {
        float4 c = 0;
        for (int j = 0; j < niter; ++j) {
          int zero = j >> 31;
          c += read_imagef(src, SAMPLER, (int2)(zero, get_global_id(0)));
        }
        dst[0] = c.x * c.y * c.z * c.w;
      }
    )";
    cl::Program program = env.create_program(src, "");
    cl::Kernel kernels[] = {
      env.create_kernel(program, "img_cacheline_size_x"),
      env.create_kernel(program, "img_cacheline_size_y"),
    };

    cl::ImageFormat img_fmt(CL_RGBA, CL_FLOAT);
    cl::Image2D in_img =
      env.create_img_2d(0, img_fmt, MAX_IMG_WIDTH, MAX_IMG_HEIGHT);
    cl::Buffer out_buf = env.create_buf(0, sizeof(float));

    auto bench = [&](uint32_t nthread, uint32_t dim) {
      archprobe::assert(dim < 2, "invalid image dimension");
      auto& kernel = kernels[dim];
      
      cl::NDRange global(nthread, 1, 1);
      cl::NDRange local(nthread, 1, 1);
      kernel.setArg(0, in_img);
      kernel.setArg(1, out_buf);
      kernel.setArg(2, (int)(NITER));
      auto time = env.bench_kernel(kernel, local, global, 200);
      env.table().push(nthread, dim, NITER, time);
      return time;
    };

    const char* report_name_by_dim[] {
      "ImgMinTimeConcurThreadCountX",
      "ImgMinTimeConcurThreadCountY",
    };

    uint32_t concur_nthread_by_dim[2];
    for (uint32_t dim = 0; dim < 2; ++dim) {
      const uint32_t IMG_EDGE = dim == 0 ? MAX_IMG_WIDTH : MAX_IMG_HEIGHT;
      const uint32_t IMG_OTHER_EDGE = dim == 1 ? MAX_IMG_WIDTH : MAX_IMG_HEIGHT;
      const uint32_t MAX_NTHREAD = std::min(NTHREAD_LOGIC, IMG_OTHER_EDGE);

      uint32_t& concur_nthread = concur_nthread_by_dim[dim];

      const char* report_name = report_name_by_dim[dim];
      if (!env.try_get_report(report_name, concur_nthread) && done) {
        log::info("testing image cacheline size along dim=", dim);

        env.ensure_min_niter(1000, NITER, [&]() {
          return bench(1, dim);
        });

        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        uint32_t nthread = 1;
        for (; nthread <= MAX_NTHREAD; ++nthread) {
          double time = bench(nthread, dim);
          log::debug("testing nthread=", nthread, ", time=", time);

          if (dj.push(time)) {
            concur_nthread = nthread - 1;
            log::info("can concurrently access ", concur_nthread, "px with "
              "minimal cost along dim=", dim);
            break;
          }
        }
        if (nthread >= MAX_NTHREAD) {
          log::warn("unable to conclude a top level image cacheline size");
          done = false;
        } else {
          concur_nthread_by_dim[dim] = concur_nthread;
          env.report_value(report_name, concur_nthread);
        }
      }
    }

    const uint32_t concur_nthread_x = concur_nthread_by_dim[0];
    const uint32_t concur_nthread_y = concur_nthread_by_dim[1];

    uint32_t cacheline_size;
    if (!env.try_get_report("ImgCachelineSize", cacheline_size) && done) {
      cacheline_size = PX_SIZE * std::max(concur_nthread_x, concur_nthread_y) /
        std::min(concur_nthread_x, concur_nthread_y);
      env.report_value("ImgCachelineSize", cacheline_size);
    }

    std::string cacheline_dim;
    if (!env.try_get_report("ImgCachelineDim", cacheline_dim) && done) {
      cacheline_dim = concur_nthread_x >= concur_nthread_y ? "X" : "Y";
      env.report_value("ImgCachelineDim", cacheline_dim);
    }

    env.report_ready(done);
  }

  // This aspect benchmarks the maximal GFLOP/s (giga-floating-point-operations
  // per second) the entire GPU could achieve with all logically concurrent
  // threads activated.
  //
  // In the experiment, we repeatedly `mad` (fused multiply-add) the operands
  // and calculate the expected number of operations, and divide it by recorded
  // time. As the iteration number increases, the initialization cost will be
  // spreaded out.
  //
  // NOTE: This aspect DOES NOT directly contribute to algorithm design but
  // allows the user to verify how much we have achieved comparing against the
  // peak performance.
  void gflops(Environment& env) {
    if (env.report_started_lazy("Gflops")) { return; }
    env.init_table("float width (bit)", "ncomp", "niter", "t (us)");
    bool done = true;

    const uint32_t NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    const uint32_t NSM =
      env.must_get_aspect_report<uint32_t>("Device", "SmCount");

    const double COMPENSATE = env.cfg_num("Compensate", 0.01);
    const double THRESHOLD = env.cfg_num("Threshold", 10);

    const uint32_t MAX_NCOMP_LOG2 = 4;
    const uint32_t MAX_NCOMP = 1 << MAX_NCOMP_LOG2;

    // In case that NTHREAD_LOGIC * NSM is not enough to saturate the processing
    // power, multiply it by a large constant.
    const uint32_t NGRP = NSM * 64;
    const uint32_t NFLOP_PER_UNROLL = 4; // Two muls and two adds per vector.
    // The following 16 is just a random large constant.
    const uint32_t NUNROLL_PER_ITER = MAX_NCOMP * 16;
    uint32_t NITER;

    cl::Buffer out_buf = env.create_buf(0, sizeof(float));

    auto bench = [&](uint32_t float_width, uint32_t ncomp) {
      const char* dtype = (float_width == 16 ? "half" : "float");
      auto dtype_vec = vec_name_by_ncomp(dtype, ncomp);

      auto src = util::format(R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
        #define DTYPE )", dtype, R"(
        #define DTYPE_VEC )", dtype_vec, R"(
        __kernel void gflops(
          __global DTYPE* out_buf
        ) {
          DTYPE_VEC x = get_local_id(0);
          DTYPE_VEC y = get_local_id(1);

          for (int i = 0; i < )", NITER, R"(; i++) {)",
        [&](){
          std::stringstream ss;
          for (auto j = 0; j < NUNROLL_PER_ITER / ncomp; ++j) {
            ss << "x = mad(y, y, x); y = mad(x, x, y);";
          }
          return ss.str();
        }(),
        R"(}
          out_buf[get_global_id(0) >> 31] = 0 )",
        [ncomp](){
          if (ncomp > 1) {
            std::stringstream ss;
            for (auto j = 0; j < ncomp; ++j) {
              char hex_digit = j < 10 ? j + '0' : j - 10 + 'A';
              ss << util::format(" + y.s", hex_digit);
            }
            return ss.str();
          } else {
            return std::string(" + y");
          }
        }(),
        R"(;
        }
      )");
      const char* build_opts = "-Werror -cl-mad-enable -cl-fast-relaxed-math";
      cl::Program program = env.create_program(src, build_opts);
      cl::Kernel kernel = env.create_kernel(program, "gflops");

      cl::NDRange global(NTHREAD_LOGIC, NGRP, 1);
      cl::NDRange local(NTHREAD_LOGIC, 1, 1);

      size_t dummy = 0;
      kernel.setArg(0, out_buf);
      auto time = env.bench_kernel(kernel, local, global, 10);
      env.table().push(float_width, ncomp, NITER, time);
      return time;
    };

    auto bench_ncomps = [&](uint32_t float_width) {
      env.ensure_min_niter(1000, NITER, [&]() {
        return bench(float_width, 16);
      });

      const uint64_t NFLOP_PER_THREAD =
        NITER * NFLOP_PER_UNROLL * NUNROLL_PER_ITER;
      const uint64_t NFLOP = NFLOP_PER_THREAD * NTHREAD_LOGIC * NGRP;

      DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);

      std::string arch;
      uint32_t ncomp;
      double gflops;
      if ((
        !env.try_get_report("Arch", arch) ||
        !env.try_get_report("VecComponentCount", ncomp) ||
        !env.try_get_report("Gflops", gflops)
      ) && done) {
        for (ncomp = 1; ncomp <= MAX_NCOMP; ncomp <<= 1) {
          auto time = bench(float_width, ncomp);

          gflops = (double)NFLOP / time * 1e-3;
          log::debug("fp", float_width, "x", ncomp, " computation throughput "
            "is ", gflops, " gflops (", time, " us)");

          if (dj.push(time)) {
            ncomp /= 2;
            log::info("optimal component count for compute for fp", float_width,
              " is ", ncomp, "which achieved ", gflops, " gflop/s at most");
            break;
          }
        }

        std::string report_prefix = float_width == 16 ? "Half" : "Float";
        if (ncomp >= MAX_NCOMP) {
          env.report_value(report_prefix + "Arch", "SISD");
          env.report_value(report_prefix + "VecComponentCount", 1);
        } else {
          env.report_value(report_prefix + "Arch", "SIMD");
          env.report_value(report_prefix + "VecComponentCount", ncomp);
        }
        env.report_value(report_prefix + "Gflops", gflops);
      }
    };

    bench_ncomps(16);
    bench_ncomps(32);

    env.report_ready(done);
  }

  // This aspect tests the preferred vectorization of data read. GPUs usually
  // prefer reading data in pack of four 32b-words because colors and position
  // vectors in graphics usually have three or four components; but GPGPUs might
  // read wider than that.
  //
  // In the experiment, we repeatedly read a vector of a varying width, with a
  // fixed stride so that the area of memory being accessed as well as cache
  // miss count can be fixed. Reading less than the preferred vectorization
  // should have the same cost as reading the entire of it.
  //
  // NOTE: Even though reading shorter vectors can reduce latency but we
  // ultimately want to minimize the average cost of reading each word. So we
  // only report the vector size when further increase in vector size no longer
  // brings better word-time efficiency.
  //
  // NOTE: We don't have to do this for images because they are DEFINED to read
  // four-component vectors by the specification.
  void buf_vec_width(Environment& env) {
    if (env.report_started_lazy("BufferVecWidth")) { return; }
    env.init_table("size (byte)", "niter", "t (us)");
    bool done = true;

    const uint32_t NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");

    const double COMPENSATE = env.cfg_num("Compensate", 0.01);
    const double THRESHOLD = env.cfg_num("Threshold", 10);

    uint32_t NITER;

    const uint32_t MAX_NCOMP_LOG2 = 4;
    const size_t PITCH = sizeof(float) << MAX_NCOMP_LOG2;

    cl::Buffer in_buf = env.create_buf(0, NTHREAD_LOGIC * PITCH);
    cl::Buffer out_buf = env.create_buf(0, PITCH);

    auto bench = [&](uint32_t ncomp) {
      std::string dtype = "float" + (ncomp == 1 ? "" : std::to_string(ncomp));

      auto src = util::format(R"(
        __kernel void buf_vec_width(
          __global const )", dtype, R"(* in_buf,
          __global )", dtype, R"(* out_buf,
          __private const int nread
        ) {
          )", dtype, R"( x = 0;
          for (int i = 0; i < nread; ++i) {
            const int zero = i >> 31;
            x += in_buf[zero];
          }
          out_buf[0] = x;
        }
      )");
      cl::Program program = env.create_program(src, "");
      cl::Kernel kernel = env.create_kernel(program, "buf_vec_width");

      cl::NDRange global(NTHREAD_LOGIC, 1, 1);
      cl::NDRange local(NTHREAD_LOGIC, 1, 1);

      kernel.setArg(0, in_buf);
      kernel.setArg(1, out_buf);
      kernel.setArg(2, (int)(NITER));
      auto time = env.bench_kernel(kernel, local, global, 10);
      env.table().push(sizeof(float) * ncomp, NITER, time);
      return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(1); });


    uint32_t vec_width_max = 0;
    if (!env.try_get_report("BufferVecSize", vec_width_max) && done) {
      DtJumpFinder<1> dj(COMPENSATE, THRESHOLD);

      uint32_t ncomp_log2 = 0;
      for (; ncomp_log2 <= MAX_NCOMP_LOG2; ++ncomp_log2) {
        uint32_t ncomp = 1 << ncomp_log2;

        double time = bench(ncomp);
        log::debug("reading ", sizeof(float) * ncomp, "B vectors takes ",
          time, "us");

        if (dj.push(time)) {
          vec_width_max = ncomp / 2;
          log::info("optimal buffer read size is ", sizeof(float) * ncomp, "B");
          break;
        }
      }
      if (ncomp_log2 >= MAX_NCOMP_LOG2) {
        log::warn("unable to conclude an optimal buffer vector size");
        done = false;
      }

      log::info("discovered the optimal vectorization for buffer "
        "access in 32b-words be ", vec_width_max);
      env.report_value("BufferVecSize", vec_width_max);
    }

    env.report_ready(done);
  }

  void img_bandwidth(Environment& env) {
    if (env.report_started_lazy("ImageBandwidth")) { return; }
    env.init_table("range (byte)", "t (us)", "bandwidth (gbps)");
    bool done = true;

    const int MAX_IMAGE_WIDTH =
      env.must_get_aspect_report<uint32_t>("Device", "MaxImageWidth");
    const int NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    const int NSM =
      env.must_get_aspect_report<uint32_t>("Device", "SmCount");

    // Size configs in bytes. These settings should be adjusted by hand.
    const uint32_t VEC_WIDTH = 4;
    const size_t VEC_SIZE = VEC_WIDTH * sizeof(float);
    const size_t RANGE = 128 * 1024 * 1024;
    const uint32_t NFLUSH = 4;
    const uint32_t NUNROLL = 16;
    const uint32_t NITER = 4;
    const uint32_t NREAD_PER_THREAD = NUNROLL * NITER;

    auto bench = [&](size_t access_size) {
      const size_t CACHE_SIZE = access_size;

      const size_t NVEC = RANGE / VEC_SIZE;
      const size_t NVEC_CACHE = CACHE_SIZE / VEC_SIZE;

      const int nthread_total = NVEC / NREAD_PER_THREAD;
      const int local_x = NTHREAD_LOGIC;
      const int global_x = (nthread_total / local_x * local_x) * NSM * NFLUSH;
      //log::debug("local_x=", local_x, "; global_x=", global_x);

      auto src = util::format(R"(
        __constant sampler_t SAMPLER =
          CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
        __kernel void buf_bandwidth(
          __read_only image1d_t A,
          __global float4 *B,
          __private const int niter,
          __private const int addr_mask
        ) {
          float4 sum = 0;
          int offset = (get_group_id(0) * )", local_x * NREAD_PER_THREAD,
          R"( + get_local_id(0)) & addr_mask;

          for (int i = 0; i < niter; ++i)
          {)", [&]() {
            std::stringstream ss;
            for (int i = 0; i < NUNROLL; ++i) {
              ss << "sum *= read_imagef(A, SAMPLER, offset); "
                "offset = (offset + " << local_x << ") & addr_mask;\n";
            }
            return ss.str();
            }(), R"(
          }
          B[get_local_id(0)] = sum;
        })");
      //log::debug(src);
      cl::Program program = env.create_program(src, "");
      cl::Kernel kernel = env.create_kernel(program, "buf_bandwidth");

      cl::ImageFormat img_fmt(CL_RGBA, CL_FLOAT);
      cl::Image1D in_buf = env.create_img_1d(0, img_fmt,
        std::min<size_t>(NVEC, env.dev_report.img_width_max));
      cl::Buffer out_buf = env.create_buf(0, VEC_SIZE * NTHREAD_LOGIC);

      cl::NDRange global(global_x, 1, 1);
      cl::NDRange local(local_x, 1, 1);

      kernel.setArg(0, in_buf);
      kernel.setArg(1, out_buf);
      kernel.setArg(2, int(NITER));
      kernel.setArg(3, int(NVEC_CACHE - 1));
      auto time = env.bench_kernel(kernel, local, global, 10);
      const size_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
      auto gbps = SIZE_TRANS * 1e-3 / time;
      log::debug("image bandwidth accessing ", access_size, "B unique data is ",
        gbps, " gbps (", time, " us)");
      env.table().push(access_size, time, gbps);
      return gbps;
    };

    MaxStats<double> max_bandwidth;
    MinStats<double> min_bandwidth;
    for (size_t access_size = VEC_SIZE; access_size < RANGE; access_size *= 2) {
      double gbps = bench(access_size);
      max_bandwidth.push(gbps);
      min_bandwidth.push(gbps);
    }

    env.report_value("MaxBandwidth", max_bandwidth);
    env.report_value("MinBandwidth", min_bandwidth);

    log::info("discovered image read bandwidth min=", (double)min_bandwidth,
      "; max=", (double)max_bandwidth);

    env.report_ready(done);
  }

  void buf_bandwidth(Environment& env) {
    if (env.report_started_lazy("BufferBandwidth")) { return; }
    env.init_table("range (byte)", "t (us)", "bandwidth (gbps)");
    bool done = true;

    const int NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    const int NSM =
      env.must_get_aspect_report<uint32_t>("Device", "SmCount");

    // Size configs in bytes. These settings should be adjusted by hand.
    const size_t RANGE = 128 * 1024 * 1024;
    const uint32_t VEC_WIDTH = 4;
    const uint32_t NFLUSH = 4;
    const uint32_t NUNROLL = 16;
    const uint32_t NITER = 4;
    const uint32_t NREAD_PER_THREAD = NUNROLL * NITER;

    const size_t VEC_SIZE = VEC_WIDTH * sizeof(float);
    auto bench = [&](size_t access_size) {
      const size_t CACHE_SIZE = access_size;

      const size_t NVEC = RANGE / VEC_SIZE;
      const size_t NVEC_CACHE = CACHE_SIZE / VEC_SIZE;

      const int nthread_total = NVEC / NREAD_PER_THREAD;
      const int local_x = NTHREAD_LOGIC;
      const int global_x = (nthread_total / local_x * local_x) * NSM * NFLUSH;
      //log::debug("local_x=", local_x, "; global_x=", global_x);

      auto src = util::format(R"(
        __kernel void buf_bandwidth(
          __global float4 *A,
          __global float4 *B,
          __private const int niter,
          __private const int addr_mask
        ) {
          float4 sum = 0;
          int offset = (get_group_id(0) * )", local_x * NREAD_PER_THREAD,
          R"( + get_local_id(0)) & addr_mask;

          for (int i = 0; i < niter; ++i)
          {)", [&]() {
            std::stringstream ss;
            for (int i = 0; i < NUNROLL; ++i) {
              ss << "sum *= A[offset]; offset = (offset + " << local_x
                << ") & addr_mask;\n";
            }
            return ss.str();
            }(), R"(
          }
          B[get_local_id(0)] = sum;
        })");
      //log::debug(src);
      cl::Program program = env.create_program(src, "");
      cl::Kernel kernel = env.create_kernel(program, "buf_bandwidth");

      cl::Buffer in_buf = env.create_buf(0, RANGE);
      cl::Buffer out_buf = env.create_buf(0, VEC_SIZE * NTHREAD_LOGIC);

      cl::NDRange global(global_x, 1, 1);
      cl::NDRange local(local_x, 1, 1);

      kernel.setArg(0, in_buf);
      kernel.setArg(1, out_buf);
      kernel.setArg(2, int(NITER));
      kernel.setArg(3, int(NVEC_CACHE - 1));
      auto time = env.bench_kernel(kernel, local, global, 10);
      const size_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
      auto gbps = SIZE_TRANS * 1e-3 / time;
      log::debug("buffer bandwidth accessing ", access_size,
        "B unique data is ", gbps, " gbps (", time, " us)");
      env.table().push(access_size, time, gbps);
      return gbps;
    };

    MaxStats<double> max_bandwidth {};
    MinStats<double> min_bandwidth {};
    for (size_t access_size = VEC_SIZE; access_size < RANGE; access_size *= 2) {
      double gbps = bench(access_size);
      max_bandwidth.push(gbps);
      min_bandwidth.push(gbps);
    }

    env.report_value("MaxBandwidth", max_bandwidth);
    env.report_value("MinBandwidth", min_bandwidth);

    log::info("discovered buffer read bandwidth min=", (double)min_bandwidth,
      "; max=", (double)max_bandwidth);

    env.report_ready(done);
  }

  void const_mem_bandwidth(Environment& env) {
    if (env.report_started_lazy("ConstMemBandwidth")) { return; }
    env.init_table("range (byte)", "t (us)", "bandwidth (gbps)");
    bool done = true;

    const int NTHREAD_WARP =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    const int NSM =
      env.must_get_aspect_report<uint32_t>("Device", "SmCount");
    const size_t RANGE =
      env.must_get_aspect_report<uint32_t>("Device", "MaxConstMemSize");

    // Size configs in bytes. These settings should be adjusted by hand.
    const uint32_t VEC_WIDTH = 4;
    const uint32_t NFLUSH = 16;
    const uint32_t NUNROLL = 16;
    const uint32_t NITER = 4;
    const uint32_t NREAD_PER_THREAD = NUNROLL * NITER;

    const size_t VEC_SIZE = VEC_WIDTH * sizeof(float);
    auto bench = [&](size_t access_size) {
      const size_t CACHE_SIZE = access_size;

      const size_t NVEC = RANGE / VEC_SIZE;
      const size_t NVEC_CACHE = CACHE_SIZE / VEC_SIZE;

      // The thread count is doesn't divide by thread workload basically because
      // of the limited memory size. Constant memory and local memory are
      // usually sub-MB level but buffer and images can go upto gigs.
      const int nthread_total = NVEC;
      const int local_x = NTHREAD_WARP;
      const int global_x = (nthread_total / local_x * local_x) * NSM * NFLUSH;
      //log::debug("local_x=", local_x, "; global_x=", global_x);

      auto src = util::format(R"(
        __kernel void const_mem_bandwidth(
          __constant float4 *A,
          __global float4 *B,
          __private const int niter,
          __private const int addr_mask
        ) {
          float4 sum = 0;
          int offset = (get_group_id(0) * )", local_x * NREAD_PER_THREAD,
          R"( + get_local_id(0)) & addr_mask;

          for (int i = 0; i < niter; ++i)
          {)", [&]() {
            std::stringstream ss;
            for (int i = 0; i < NUNROLL; ++i) {
              ss << "sum *= A[offset]; offset = (offset + " << local_x
                << ") & addr_mask;\n";
            }
            return ss.str();
            }(), R"(
          }
          B[get_local_id(0)] = sum;
        })");
      //log::debug(src);
      cl::Program program = env.create_program(src, "");
      cl::Kernel kernel = env.create_kernel(program, "const_mem_bandwidth");

      cl::Buffer in_buf = env.create_buf(0, CACHE_SIZE);
      cl::Buffer out_buf = env.create_buf(0, VEC_SIZE * NTHREAD_WARP);

      cl::NDRange global(global_x, 1, 1);
      cl::NDRange local(local_x, 1, 1);

      kernel.setArg(0, in_buf);
      kernel.setArg(1, out_buf);
      kernel.setArg(2, int(NITER));
      kernel.setArg(3, int(NVEC_CACHE - 1));
      auto time = env.bench_kernel(kernel, local, global, 10);
      const size_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
      auto gbps = SIZE_TRANS * 1e-3 / time;
      log::debug("constant memory bandwidth accessing ", access_size,
        "B unique data is ", gbps, " gbps (", time, " us)");
      env.table().push(access_size, time, gbps);
      return gbps;
    };

    MaxStats<double> max_bandwidth {};
    MinStats<double> min_bandwidth {};
    for (size_t access_size = VEC_SIZE; access_size < RANGE; access_size *= 2) {
      double gbps = bench(access_size);
      max_bandwidth.push(gbps);
      min_bandwidth.push(gbps);
    }

    env.report_value("MaxBandwidth", max_bandwidth);
    env.report_value("MinBandwidth", min_bandwidth);

    log::info("discovered constant memory read bandwidth min=",
      (double)min_bandwidth, "; max=", (double)max_bandwidth);

    env.report_ready(done);
  }

  void local_mem_bandwidth(Environment& env) {
    if (env.report_started_lazy("LocalMemBandwidth")) { return; }
    env.init_table("range (byte)", "t (us)", "bandwidth (gbps)");
    bool done = true;

    const int NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");
    const int NSM =
      env.must_get_aspect_report<uint32_t>("Device", "SmCount");
    const size_t RANGE =
      env.must_get_aspect_report<uint32_t>("Device", "MaxLocalMemSize");

    // Size configs in bytes. These settings should be adjusted by hand.
    const uint32_t VEC_WIDTH = 4;
    const uint32_t NFLUSH = 16;
    const uint32_t NUNROLL = 16;
    const uint32_t NITER = 4;
    const uint32_t NREAD_PER_THREAD = NUNROLL * NITER;

    const size_t VEC_SIZE = VEC_WIDTH * sizeof(float);
    auto bench = [&](size_t access_size) {
      const size_t CACHE_SIZE = access_size;

      const size_t NVEC = RANGE / VEC_SIZE;
      const size_t NVEC_CACHE = CACHE_SIZE / VEC_SIZE;

      const int nthread_total = NVEC;
      const int local_x = NTHREAD_LOGIC;
      const int global_x = (nthread_total / local_x * local_x) * NSM * NFLUSH;
      //log::debug("local_x=", local_x, "; global_x=", global_x);

      auto src = util::format(R"(
        __kernel void local_mem_bandwidth(
          __global float4 *B,
          __private const int niter,
          __private const int addr_mask
        ) {
          __local float4 A[)", CACHE_SIZE / VEC_SIZE, R"(];
          A[get_local_id(0)] = get_local_id(0);
          barrier(CLK_LOCAL_MEM_FENCE);

          float4 sum = 0;
          int offset = (get_group_id(0) * )", local_x * NREAD_PER_THREAD,
          R"( + get_local_id(0)) & addr_mask;

          for (int i = 0; i < niter; ++i)
          {)", [&]() {
            std::stringstream ss;
            for (int i = 0; i < NUNROLL; ++i) {
              ss << "sum *= A[offset]; offset = (offset + " << local_x
                << ") & addr_mask;\n";
            }
            return ss.str();
            }(), R"(
          }
          B[get_local_id(0)] = sum;
        })");
      //log::debug(src);
      cl::Program program = env.create_program(src, "");
      cl::Kernel kernel = env.create_kernel(program, "local_mem_bandwidth");

      cl::Buffer out_buf = env.create_buf(0, VEC_SIZE * NTHREAD_LOGIC);

      cl::NDRange global(global_x, 1, 1);
      cl::NDRange local(local_x, 1, 1);

      kernel.setArg(0, out_buf);
      kernel.setArg(1, int(NITER));
      kernel.setArg(2, int(NVEC_CACHE - 1));
      auto time = env.bench_kernel(kernel, local, global, 10);
      const size_t SIZE_TRANS = global_x * NREAD_PER_THREAD * VEC_SIZE;
      auto gbps = SIZE_TRANS * 1e-3 / time;
      log::debug("local memory bandwidth accessing ", access_size,
        "B unique data is ", gbps, " gbps (", time, " us)");
      env.table().push(access_size, time, gbps);
      return gbps;
    };

    MaxStats<double> max_bandwidth {};
    MinStats<double> min_bandwidth {};
    for (size_t access_size = VEC_SIZE; access_size < RANGE; access_size *= 2) {
      double gbps = bench(access_size);
      max_bandwidth.push(gbps);
      min_bandwidth.push(gbps);
    }

    env.report_value("MaxBandwidth", max_bandwidth);
    env.report_value("MinBandwidth", min_bandwidth);

    log::info("discovered local memory read bandwidth min=",
      (double)min_bandwidth, "; max=", (double)max_bandwidth);

    env.report_ready(done);
  }

  // This aspect tests the warping of the SMs. A warp is an atomic schedule unit
  // where all threads in a warp can be executed in parallel. An GPU SM usually
  // consumes more threads than it physically can so that it can hide the
  // latency of expensive operations like memory accesses by interleaved
  // execution.
  //
  // The warping mechanism is, however, hard to conclude with a single method
  // because of two kinds of scheduling behavior the driver could have:
  //
  // 1. It waits for more threads than it physically have. (ARM Mali)
  // 2. It runs more threads than it logically should. (Qualcomm Adreno)
  //
  // In Case 1, the SM waits until all the works and dummy works to finish, for
  // a uniform control flow limited by a shared program counter. Even if a
  // number of threads are not effective workloads, the SM cannot exit early and
  // still have to spend time executing the dummy works.
  //
  // In Case 2, the SM acquires a large number of threads and the threads are
  // distributed to physical threads in warps. When the incoming workload has
  // too few threads. The driver might decides to pack multiple works together
  // and dispatch them at once.
  //
  // We devised two different micro-benchmarks to reveal the true parallelism
  // capacity of the underlying platform against the interference of these
  // situations.
  void warp_size(Environment& env) {
    const uint32_t NTHREAD_LOGIC =
      env.must_get_aspect_report<uint32_t>("Device", "LogicThreadCount");

    // Method A: Let all the threads in a warp to race and atomically fetch-add
    // a counter, then store the counter values to the output buffer as
    // scheduling order of these threads. If all the order numbers followings a
    // pattern charactorized by the scheduler hardware, then the threads are
    // likely executing within a warp. Threads in different warps are not
    // managed by the same scheduler, so they would racing for a same ID out of
    // order, unaware of each other.
    //
    // This method helps us identify warp sizes when the SM sub-divides its ALUs
    // into independent groups, like the three execution engines in a Mali G76
    // core. It helps warp-probing in Case 1 because it doesn't depend on kernel
    // timing, so the extra wait time doesn't lead to inaccuracy.
    if (!env.report_started_lazy("WarpSizeMethodA")) {
      env.init_table("nthread", "nascend");
      bool done = true;

      int nthread_warp_a;
      if (!env.try_get_report("WarpThreadCount", nthread_warp_a) && done) {
        auto src = R"(
          __kernel void warp_size(__global int* output) {
            __local int local_counter;
            local_counter = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            int i = atomic_inc(&local_counter);
            barrier(CLK_LOCAL_MEM_FENCE);
            output[get_global_id(0)] = i;
          }
        )";

        cl::Program program = env.create_program(src, "");
        cl::Kernel kernel = env.create_kernel(program, "warp_size");

        auto bench = [&](uint32_t nthread) {
          cl::NDRange global(nthread, 1, 1);
          cl::NDRange local(nthread, 1, 1);

          auto size = NTHREAD_LOGIC * sizeof(int32_t);
          cl::Buffer out_buf = env.create_buf(0, size);

          kernel.setArg(0, out_buf);
          env.bench_kernel(kernel, local, global, 1);

          int32_t nascend;
          auto mapped = env.map_buf(out_buf);
          {
            int32_t* data = (int32_t*)(void*)mapped;

            std::stringstream ss;
            for (auto j = 0; j < nthread; ++j) { ss << data[j] << " "; }
            log::debug(ss.str());

            int32_t last = -1;
            auto j = 0;
            for (; j < nthread; ++j) {
              if (last >= data[j]) { break; }
              last = data[j];
            }
            nascend = j;
          }
          env.unmap_buf(out_buf, mapped);
          env.table().push(nthread, nascend);
          return nascend;
        };

        // TODO: (penguinliong) Improve this warp size inference by stats of all
        // sequential ascend of order of multiple executions.
        int i = 1;
        for (; i <= NTHREAD_LOGIC; ++i) {
          uint32_t nascend = bench(i);
          if (nascend != i) {
            nthread_warp_a = i - 1;
            break;
          }
        }
        if (i > NTHREAD_LOGIC) {
          log::warn("unable to conclude a warp size by method a");
          done = false;
        }
        env.report_value("WarpThreadCount", nthread_warp_a);
      }
      env.report_ready(done);
    }

    // Method B: Time the latency of a pure computation kernel, with different
    // number of logical threads in a workgroup. When the logical thread count
    // exceeds the physical capability of the GPU, kernel latency jumps slightly
    // to indicate higher scheduling cost.
    //
    // This timing-based method helps us identify warp sizes in Case 2, when
    // threads of multiple warps are managed by the same scheduler at the same
    // time.
    //
    // The division in the kernel helps to reduce the impact of latency hiding
    // by warping because integral division is an examplary multi-cycle
    // instruction that can hardly be optimized.
    int nthread_warp_b;
    if (!env.report_started_lazy("WarpSizeMethodB")) {
      env.init_table("nthread", "time (us)");
      bool done = true;

      const int PRIME = 3;
      const double COMPENSATE = env.cfg_num("Compensate", 0.01);
      const double THRESHOLD = env.cfg_num("Threshold", 10);

      uint32_t NITER;

      auto src = R"(
        __kernel void warp_size2(
          __global float* src,
          __global int* dst,
          const int niter,
          const int prime_number
        ) {
          int drain = 0;
          for (int j = 0; j < niter; ++j) {
            drain += j / prime_number;
            barrier(0);
          }
          dst[get_local_id(0)] = drain;
        }
      )";
      cl::Program program = env.create_program(src, "");
      cl::Kernel kernel = env.create_kernel(program, "warp_size2");

      auto size = NTHREAD_LOGIC * sizeof(float);
      cl::Buffer src_buf = env.create_buf(0, size);
      cl::Buffer dst_buf = env.create_buf(0, size);

      auto bench = [&](uint32_t nthread) {
        kernel.setArg(0, src_buf);
        kernel.setArg(1, dst_buf);
        kernel.setArg(2, NITER);
        kernel.setArg(3, PRIME);

        cl::NDRange global(nthread, 1024, 1);
        cl::NDRange local(nthread, 1, 1);

        double time = env.bench_kernel(kernel, local, global, 10);
        env.table().push(nthread, time);
        return time;
      };

      env.ensure_min_niter(1000, NITER, [&]() { return bench(1); });

      DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
      auto nthread = 1;
      for (; nthread <= NTHREAD_LOGIC; ++nthread) {
        double time = bench(nthread);
        log::debug("nthread=", nthread, " (", time, "us)");
        if (dj.push(time)) {
          nthread_warp_b = nthread - 1;
          env.report_value("WarpThreadCount", nthread_warp_b);
          break;
        }
      }
      if (nthread >= NTHREAD_LOGIC) {
        log::warn("unable to conclude a warp size by method b");
        done = false;
      }

      log::info("discovered the warp size being ", nthread_warp_b,
        " by method b");

      env.report_ready(done);
    }


  }


  void img_cache_hierarchy_pchase(Environment& env) {
    if (env.report_started_lazy("ImageCacheHierarchyPChase")) { return; }
    env.init_table("range (byte)", "stride (byte)", "niter", "t (us)");
    bool done = true;

    const uint32_t MAX_IMG_WIDTH =
      env.must_get_aspect_report<uint32_t>("Device", "MaxImageWidth");
    const uint32_t NCOMP = 4;
    const uint32_t PX_SIZE = NCOMP * sizeof(int32_t);

    static_assert(NCOMP == 1 || NCOMP == 2 || NCOMP == 4,
      "image component count must be 1, 2 or 4");

    const uint32_t MAX_DATA_SIZE =
      env.cfg_num("DataSizeMax", MAX_IMG_WIDTH * PX_SIZE);
    const uint32_t NPX = MAX_DATA_SIZE / PX_SIZE;

    const uint32_t MAX_LV = 4;

    // Compensate less because p-chase is much more sensitive than other tests.
    double COMPENSATE = env.cfg_num("Compensate", 0.01);
    const double THRESHOLD = env.cfg_num("Threshold", 10);

    uint32_t NITER;

    cl::Buffer dst_buf = env.create_buf(0, sizeof(int32_t));

    auto src = util::format(R"(
      __constant sampler_t SAMPLER =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      __kernel void img_cache_hierarchy_pchase(
        __read_only image1d_t src,
        __global int* dst,
        const int niter
      ) {
        int idx = 0;
        for (int i = 0; i < niter; ++i) {
          idx = read_imagei(src, SAMPLER, idx).x;
        }
        *dst = idx;
      }
    )");
    cl::Program program = env.create_program(src, "");
    cl::Kernel kernel =
      env.create_kernel(program, "img_cache_hierarchy_pchase");

    cl::ImageFormat img_fmt(channel_order_by_ncomp(NCOMP), CL_SIGNED_INT32);
    cl::Image1D src_img = env.create_img_1d(0, img_fmt, NPX);

    auto bench = [&](uint32_t ndata, uint32_t stride) {
      auto mapped = env.map_img_1d(src_img);
      int32_t* idx_buf = (int32_t*)(void*)mapped;
      // The loop ends at `ndata` because we don't expect to read more than this
      // amount.
      for (uint32_t i = 0; i < ndata; ++i) {
        idx_buf[i * NCOMP] = (i + stride) % ndata;
      }
      env.unmap_img_1d(src_img, mapped);

      cl::NDRange global(1, 1, 1);
      cl::NDRange local(1, 1, 1);

      kernel.setArg(0, src_img);
      kernel.setArg(1, dst_buf);
      kernel.setArg(2, NITER);

      double time = env.bench_kernel(kernel, local, global, 10);
      log::debug("range=", ndata * PX_SIZE, "B; stride=", stride * PX_SIZE,
        "B; time=", time, "us");
      env.table().push(ndata * PX_SIZE, stride * PX_SIZE, NITER, time);
      return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(1, 1); });
    // Make sure every part of the memory is accessed at least once.
    NITER = std::max(NPX * 2, NITER);
    COMPENSATE *= 1000 / bench(1, 1);

    uint32_t ndata_cacheline;
    uint32_t ndata_cache;

    ndata_cacheline = 1;
    ndata_cache = 1;

    for (uint32_t lv = 1; lv <= MAX_LV; ++lv) {

      // Step 1: Find the size of cache.
      std::string cache_name = "CachePixelCountLv" + std::to_string(lv);
      if (!env.try_get_report(cache_name, ndata_cache) && done) {
        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        uint32_t ndata = ndata_cache + ndata_cacheline;
        for (; ndata <= NPX; ndata += ndata_cacheline) {
          double time = bench(ndata, ndata_cacheline);
          if (dj.push(time)) {
            ndata_cache = ndata - ndata_cacheline;
            log::info("found cache size ",
              pretty_data_size(ndata_cache * PX_SIZE));
            env.report_value(cache_name, ndata_cache * PX_SIZE);
            break;
          }
        }
        if (ndata >= NPX) {
          ndata_cache = NPX;
          log::info("found allocation boundary ",
            pretty_data_size(ndata_cache * PX_SIZE));
          break;
        }
      }

      // Step 2: Find the size of cacheline. It's possible the test might fail
      // because of noisy benchmark results when execution time is too long.
      std::string cacheline_name = "CachelinePixelCountLv" + std::to_string(lv);
      if (!env.try_get_report(cacheline_name, ndata_cacheline) && done) {
        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        uint32_t stride = ndata_cacheline;
        for (; stride < ndata_cache; ++stride) {
          double time = bench(ndata_cache + ndata_cacheline, stride);
          if (dj.push(time)) {
            ndata_cacheline = stride - 1;
            log::info("found cacheline size ",
              pretty_data_size(ndata_cacheline * PX_SIZE));
            env.report_value(cacheline_name, ndata_cacheline * PX_SIZE);
            break;
          }
        }
      }

    }

    env.report_ready(done);
  }

  void buf_cache_hierarchy_pchase(Environment& env) {
    if (env.report_started_lazy("BufferCacheHierarchyPChase")) { return; }
    env.init_table("range (byte)", "stride (byte)", "niter", "t (us)");
    bool done = true;

    const size_t MAX_BUF_SIZE = 512 * 1024;
    const uint32_t NCOMP = 4;
    const uint32_t VEC_SIZE = NCOMP * sizeof(int32_t);

    static_assert(NCOMP == 1 || NCOMP == 2 || NCOMP == 4 || NCOMP == 8 ||
      NCOMP == 16, "buffer vector component count must be 1, 2, 4, 8 or 16");

    const uint32_t MAX_DATA_SIZE =
      env.cfg_num("DataSizeMax", MAX_BUF_SIZE * VEC_SIZE);
    const uint32_t NVEC = MAX_BUF_SIZE / VEC_SIZE;

    const uint32_t MAX_LV = 4;

    // Compensate less because p-chase is much more sensitive than other tests.
    double COMPENSATE = env.cfg_num("Compensate", 0.01);
    const double THRESHOLD = env.cfg_num("Threshold", 10);

    uint32_t NITER;

    cl::Buffer dst_buf = env.create_buf(0, sizeof(int32_t));

    auto src = util::format(R"(
      __constant sampler_t SAMPLER =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
      __kernel void buf_cache_hierarchy_pchase(
        __global )", vec_name_by_ncomp("int", NCOMP) , R"(* src,
        __global int* dst,
        const int niter
      ) {
        int idx = 0;
        for (int i = 0; i < niter; ++i) {
          idx = src[idx].x;
        }
        *dst = idx;
      }
    )");
    cl::Program program = env.create_program(src, "");
    cl::Kernel kernel =
      env.create_kernel(program, "buf_cache_hierarchy_pchase");

    cl::Buffer src_buf = env.create_buf(0, NVEC * VEC_SIZE);

    auto bench = [&](uint32_t ndata, uint32_t stride) {
      auto mapped = env.map_buf(src_buf);
      int32_t* idx_buf = (int32_t*)(void*)mapped;
      for (uint64_t i = 0; i < ndata; ++i) {
        idx_buf[i * NCOMP] = (i + stride) % ndata;
      }
      env.unmap_buf(src_buf, mapped);

      cl::NDRange global(1, 1, 1);
      cl::NDRange local(1, 1, 1);

      kernel.setArg(0, src_buf);
      kernel.setArg(1, dst_buf);
      kernel.setArg(2, NITER);

      double time = env.bench_kernel(kernel, local, global, 10);
      log::debug("range=", ndata * VEC_SIZE, "B; stride=", stride * VEC_SIZE,
        "B; time=", time, " us");
      env.table().push(ndata * VEC_SIZE, stride * VEC_SIZE, NITER, time);
      return time;
    };

    env.ensure_min_niter(1000, NITER, [&]() { return bench(1, 1); });
    // Make sure every part of the memory is accessed at least once.
    NITER = std::max(NVEC * 2, NITER);
    COMPENSATE *= 1000 / bench(1, 1);

    uint32_t ndata_cacheline;
    uint32_t ndata_cache;

    ndata_cacheline = 1;
    ndata_cache = 1;

    for (uint32_t lv = 1; lv <= MAX_LV; ++lv) {

      // Step 1: Find the size of cache.
      std::string cache_name = "CacheVectorCountLv" + std::to_string(lv);
      if (!env.try_get_report(cache_name, ndata_cache) && done) {
        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        uint32_t ndata = ndata_cache + ndata_cacheline;
        for (; ndata <= NVEC; ndata += ndata_cacheline) {
          double time = bench(ndata, ndata_cacheline);
          if (dj.push(time)) {
            ndata_cache = ndata - ndata_cacheline;
            log::info("found cache size ",
              pretty_data_size(ndata_cache * VEC_SIZE));
            env.report_value(cache_name, ndata_cache * VEC_SIZE);
            break;
          }
        }
        if (ndata >= NVEC) {
          ndata_cache = NVEC;
          log::info("found allocation boundary ",
            pretty_data_size(ndata_cache * VEC_SIZE));
          break;
        }
      }

      // Step 2: Find the size of cacheline. It's possible the test might fail
      // because of noisy benchmark results when execution time is too long
      std::string cacheline_name = "CachelineVectorCountLv" +
        std::to_string(lv);
      if (!env.try_get_report(cacheline_name, ndata_cacheline) && done) {
        DtJumpFinder<5> dj(COMPENSATE, THRESHOLD);
        uint32_t stride = ndata_cacheline;
        for (; stride < ndata_cache; ++stride) {
          double time = bench(ndata_cache + ndata_cacheline, stride);
          if (dj.push(time)) {
            ndata_cacheline = stride - 1;
            log::info("found cacheline size ",
              pretty_data_size(ndata_cacheline * VEC_SIZE));
            env.report_value(cacheline_name, ndata_cacheline * VEC_SIZE);
            break;
          }
        }
      }

    }

    env.report_ready(done);
  }





} // namespace aspects



static std::unique_ptr<ArchProbe> APP = nullptr;

void guarded_main(const std::string& clear_aspect) {
  cl_int err;

  archprobe::initialize();

  APP = std::make_unique<ArchProbe>(0);
  APP->clear_aspect_report(clear_aspect);
  (*APP)
    //.with_aspect(aspects::warp_size)
    //.with_aspect(aspects::gflops)
    //.with_aspect(aspects::reg_count)
    //.with_aspect(aspects::buf_vec_width)
    //.with_aspect(aspects::img_cacheline_size)
    //.with_aspect(aspects::buf_cacheline_size)
    //.with_aspect(aspects::img_bandwidth)
    //.with_aspect(aspects::buf_bandwidth)
    .with_aspect(aspects::const_mem_bandwidth)
    .with_aspect(aspects::local_mem_bandwidth)
    //.with_aspect(aspects::img_cache_hierarchy_pchase)
    //.with_aspect(aspects::buf_cache_hierarchy_pchase)
  ;

  APP.reset();
}

void sigproc(int sig) {
  const char* sig_name = "UNKNOWN SIGNAL";
#ifdef _WIN32
#define SIGHUP 1
#define SIGQUIT 3
#define SIGTRAP 5
#define SIGKILL 9
#endif
  switch (sig) {
    // When you interrupt adb, and adb kills ArchProbe in its SIGINT process.
    case SIGHUP: sig_name = "SIGHUP"; break;
    // When you interrupt in an `adb shell` session.
    case SIGINT: sig_name = "SIGINT"; break;
    // Other weird cases.
    case SIGQUIT: sig_name = "SIGQUIT"; break;
    case SIGTRAP: sig_name = "SIGTRAP"; break;
    case SIGABRT: sig_name = "SIGABRT"; break;
    case SIGTERM: sig_name = "SIGTERM"; break;
    case SIGKILL: sig_name = "SIGKILL"; break;
  }
  log::error("captured ", sig_name, "! progress is saved");
  APP.reset();
  std::exit(1);
}

} // namespace archprobe


struct AppConfig {
  bool verbose;
  std::string clear_aspect;
};


AppConfig configurate(int argc, const char** argv) {
  using namespace archprobe::args;
  AppConfig cfg {};
  init_arg_parse("ArchProbe",
    "Discover hardware details by micro-benchmarks.");
  reg_arg<SwitchParser>("-v", "--verbose", cfg.verbose,
    "Print more detail for debugging.");
  reg_arg<StringParser>("-c", "--clear", cfg.clear_aspect,
    "Clear the results of specified aspect.");
  parse_args(argc, argv);
  return cfg;
}

int main(int argc, const char** argv) {
  log::set_log_callback(log_cb);
  AppConfig cfg = configurate(argc, argv);
  if (cfg.verbose) {
    log::set_log_filter_level(log::LogLevel::L_LOG_LEVEL_DEBUG);
  } else {
    log::set_log_filter_level(log::LogLevel::L_LOG_LEVEL_INFO);
  }

  signal(SIGHUP, archprobe::sigproc);
  signal(SIGINT, archprobe::sigproc);
  signal(SIGQUIT, archprobe::sigproc);
  signal(SIGTRAP, archprobe::sigproc);
  signal(SIGABRT, archprobe::sigproc);
  signal(SIGTERM, archprobe::sigproc);
  signal(SIGKILL, archprobe::sigproc);

  try {
    archprobe::guarded_main(cfg.clear_aspect);
  } catch (const std::exception& e) {
    log::error("application threw an exception");
    log::error(e.what());
    log::error("application cannot continue");
  } catch (...) {
    log::error("application threw an illiterate exception");
  }

  return 0;
}
