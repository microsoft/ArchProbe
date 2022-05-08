// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "env.hpp"

namespace archprobe {

std::string pretty_data_size(size_t size) {
  const size_t K = 1024;
  if (size < K) { return util::format(size,  "B"); } size /= K;
  if (size < K) { return util::format(size, "KB"); } size /= K;
  if (size < K) { return util::format(size, "MB"); } size /= K;
  if (size < K) { return util::format(size, "GB"); } size /= K;
  if (size < K) { return util::format(size, "TB"); } size /= K;
  archprobe::panic("unsupported data size");
  return {};
}

DeviceReport collect_dev_report(const cl::Device& dev) {
  DeviceReport dev_report {};
  log::info("set-up testing environment");

  // General memory detail.
  dev_report.has_page_size = CL_SUCCESS ==
    dev.getInfo(CL_DEVICE_PAGE_SIZE_QCOM, &dev_report.page_size);
  // Global memory detail.
  dev_report.buf_cacheline_size =
    dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
  dev_report.buf_size_max = dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  dev_report.buf_cache_size = dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
  // Special memory detail.
  dev_report.const_mem_size_max =
    dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
  dev_report.local_mem_size_max =
    dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  // Image memory detail.
  dev_report.support_img = dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
  if (dev_report.support_img) {
    dev_report.img_width_max = dev.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
    dev_report.img_height_max = dev.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
  }
  // Processor detail.
  dev_report.nsm = dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  dev_report.nthread_logic = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

  log::info("fetched device report");
  log::push_indent();
  {
    if (dev_report.has_page_size) {
      log::info("(qualcomm extension) device page size is ",
        pretty_data_size(dev_report.page_size));
    }
    log::info(pretty_data_size(dev_report.buf_size_max),
      " global memory with ", pretty_data_size(dev_report.buf_cache_size),
      " cache consists of ", pretty_data_size(dev_report.buf_cacheline_size),
      " cachelines");
    if (dev_report.support_img) {
      log::info("images up to [", dev_report.img_width_max, ", ",
        dev_report.img_height_max, "] texels are supported");
    } else {
      log::info("image is not supported");
    }
    log::info(dev_report.nsm, " SMs with ", dev_report.nthread_logic,
      " logical threads in each");
  }
  log::pop_indent();

  return dev_report;
}

json::JsonValue load_env_cfg(const char* path) {
  try {
    auto json_txt = util::load_text(path);
    log::debug("loaded configuration '", json_txt, "'");
    json::JsonValue out {};
    if (json::try_parse(json_txt, out)) {
      archprobe::assert(out.is_obj());
      return out;
    } else {
      log::warn("failed to parse environment config from '", path,
        "', a default configuration will be created to overwrite it");
      return json::JsonObject {};
    }
  } catch (archprobe::AssertionFailedException) {
    log::warn("configuration file cannot be opened at '", path,
      "', a default configuration will be created");
    return json::JsonObject {};
  }
}

json::JsonValue load_report(const char* path) {
  try {
    auto json_txt = util::load_text(path);
    log::debug("loaded report '", json_txt, "'");
    json::JsonValue out {};
    if (json::try_parse(json_txt, out)) {
      archprobe::assert(out.is_obj());
      return out;
    } else {
      log::warn("failed to parse report from '", path, "', a new report "
        "will be created to overwrite it");
      return json::JsonObject {};
    }
  } catch (archprobe::AssertionFailedException) {
    log::warn("report file cannot be opened at '", path, "', a new "
      "report will be created");
    return json::JsonObject {};
  }
}


void report_dev(Environment& env) {
  if (env.report_started_lazy("Device")) { return; }
  env.report_value("SmCount", env.dev_report.nsm);
  env.report_value("LogicThreadCount", env.dev_report.nthread_logic);
  env.report_value("MaxBufferSize", env.dev_report.buf_size_max);
  env.report_value("MaxConstMemSize", env.dev_report.const_mem_size_max);
  env.report_value("MaxLocalMemSize", env.dev_report.local_mem_size_max);
  env.report_value("CacheSize", env.dev_report.buf_cache_size);
  env.report_value("CachelineSize", env.dev_report.buf_cacheline_size);
  if (env.dev_report.support_img) {
    env.report_value("MaxImageWidth", env.dev_report.img_width_max);
    env.report_value("MaxImageHeight", env.dev_report.img_height_max);
  }
  if (env.dev_report.has_page_size) {
    env.report_value("PageSize_QCOM", env.dev_report.page_size);
  }
  env.report_ready(true);
}


Environment::Environment(
  uint32_t idev,
  const char* cfg_path,
  const char* report_path
) :
  dev_(archprobe::select_dev(idev)),
  ctxt_(archprobe::create_ctxt(dev_)),
  cmd_queue_(archprobe::create_cmd_queue(ctxt_)),
  aspects_started_(),
  cur_aspect_(),
  cur_table_(nullptr),
  cfg_path_(cfg_path),
  report_path_(report_path),
  cfg_(load_env_cfg(cfg_path)),
  report_(load_report(report_path)),
  dev_report(collect_dev_report(dev_)),
  my_report()
{
  report_dev(*this);
}
Environment::~Environment() {
  util::save_text(cfg_path_.c_str(), json::print(cfg_));
  log::info("saved configuration to '", cfg_path_, "'");
  util::save_text(report_path_.c_str(), json::print(report_));
  log::info("saved report to '", report_path_, "'");
}


void Environment::report_started(const std::string& aspect_name) {
  archprobe::assert(!aspect_name.empty(), "aspect name cannot be empty");
  aspects_started_.insert(aspect_name);
  log::info("[", aspect_name, "]");
  log::push_indent();
  cur_aspect_ = aspect_name;
}
bool Environment::report_started_lazy(const std::string& aspect_name) {
  auto aspect_it = report_.obj.find(aspect_name);
  if (aspect_it == report_.obj.end() || !aspect_it->second.is_obj()) {
    report_started(aspect_name);
    return false;
  }
  auto done_it = aspect_it->second.obj.find("Done");
  if (done_it == aspect_it->second.obj.end() || !done_it->second.is_bool()) {
    report_started(aspect_name);
    return false;
  }
  if (done_it->second.b) {
    log::info("ignored aspect '", aspect_name ,"' because it's done");
    return true;
  } else {
    report_started(aspect_name);
    return false;
  }
}
void Environment::report_ready(bool done) {
  archprobe::assert(!cur_aspect_.empty(),
    "announcing ready for an not-yet-started report is not allowed");
  archprobe::assert(aspects_started_.find(cur_aspect_) != aspects_started_.end(),
    "aspect has not report to start yet");
  report_value("Done", done);
  if (cur_table_ != nullptr) {
    auto csv = cur_table_->to_csv();
    auto fname = util::format(cur_aspect_, ".csv");
    util::save_text(fname.c_str(), csv);
    log::info("saved data table to '", fname, "'");
    cur_table_ = nullptr;
  }
  cur_aspect_ = {};
  log::pop_indent();
}
void Environment::check_dep(const std::string& aspect_name) {
  bool done = false;
  archprobe::assert(try_get_aspect_report(aspect_name, "Done", done) && done,
    "aspect '", aspect_name, "' is required but is not ready yet");
}


table::Table& Environment::table() {
  archprobe::assert(cur_table_ != nullptr, "requested table is not initialized");
  return *cur_table_;
}


// Find the minimal number of iterations that a kernel can run up to
// `min_time_us` microseconds.
void Environment::ensure_min_niter(
  double min_time_us,
  uint32_t& niter,
  std::function<double()> run
) {
  const uint32_t DEFAULT_NITER = 100;
  niter = DEFAULT_NITER;
  for (uint32_t i = 0; i < 100; ++i) {
    double t = run();
    if (t > min_time_us * 0.99) {
      log::info("found minimal niter=", niter, " to take ", min_time_us,
        "us");
      return;
    }
    log::debug("niter=", niter, " doesn't run long enough (", t,
      "us <= ", min_time_us, "us)");
    niter = uint32_t(niter * min_time_us / t);
  }
  archprobe::panic("unable to find a minimal iteration number for ",
    cur_aspect_, "; is your code aggresively optimized by the compiler?");
}

} // namespace archprobe
