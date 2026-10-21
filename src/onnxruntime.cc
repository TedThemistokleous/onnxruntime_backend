// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "onnxruntime_loader.h"
#include "onnxruntime_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/backend/device_memory_tracker.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_ENABLE_ROCM
#include <hip/hip_runtime_api.h>
#endif  // TRITON_ENABLE_ROCM


//
// ONNX Runtime Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace onnxruntime {

/// Deleter for OrtSession.
struct SessionDeleter {
  void operator()(OrtSession* f) { OnnxLoader::UnloadSession(f); }
};

// Bounded queue served by its own worker threads. Push blocks while the queue
// is full, so a slow consumer applies backpressure instead of growing memory.
// Stop lets the workers finish everything already queued before they exit.
template <typename T>
class WorkQueue {
 public:
  ~WorkQueue() { Stop(); }

  void Start(
      size_t thread_count, size_t max_depth, std::function<void(T&)> fn)
  {
    max_depth_ = max_depth;
    fn_ = std::move(fn);
    for (size_t i = 0; i < thread_count; ++i) {
      threads_.emplace_back([this] { Loop(); });
    }
  }

  bool Running() const { return !threads_.empty(); }

  void Push(T&& item)
  {
    {
      std::unique_lock<std::mutex> lk(mu_);
      space_cv_.wait(lk, [this] { return q_.size() < max_depth_; });
      q_.push_back(std::move(item));
    }
    cv_.notify_one();
  }

  void Stop()
  {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      thread.join();
    }
    threads_.clear();
  }

 private:
  void Loop()
  {
    for (;;) {
      T item;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
        if (q_.empty()) {
          return;
        }
        item = std::move(q_.front());
        q_.pop_front();
      }
      space_cv_.notify_one();
      fn_(item);
    }
  }

  std::function<void(T&)> fn_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable space_cv_;
  std::deque<T> q_;
  size_t max_depth_{1};
  bool stop_{false};
};

// Runs a fixed number of parts per call: part 0 on the calling thread and
// the rest on persistent workers that sleep between calls, so a call costs
// one wake-up rather than an allocation or queue push per part.
class ForkJoin {
 public:
  ~ForkJoin() { Stop(); }

  void Start(size_t parts, std::function<void(size_t)> fn)
  {
    parts_ = std::max<size_t>(parts, 1);
    fn_ = std::move(fn);
    for (size_t part = 1; part < parts_; ++part) {
      workers_.emplace_back([this, part] { Loop(part); });
    }
  }

  void Run()
  {
    {
      std::lock_guard<std::mutex> lk(mu_);
      ++generation_;
      pending_ = parts_ - 1;
    }
    cv_.notify_all();
    fn_(0);
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this] { return pending_ == 0; });
  }

  void Stop()
  {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
    workers_.clear();
  }

 private:
  void Loop(size_t part)
  {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
      }
      fn_(part);
      bool last;
      {
        std::lock_guard<std::mutex> lk(mu_);
        last = (--pending_ == 0);
      }
      if (last) {
        done_cv_.notify_one();
      }
    }
  }

  std::function<void(size_t)> fn_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  size_t parts_{1};
  uint64_t generation_{0};
  size_t pending_{0};
  bool stop_{false};
};

// BackendConfiguration
struct BackendConfiguration {
  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_Backend* backend)
  {
    void* state = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_BackendState(backend, &state));
    return *reinterpret_cast<BackendConfiguration*>(state);
  }

  static const BackendConfiguration& RetrieveFrom(TRITONBACKEND_Model* model)
  {
    TRITONBACKEND_Backend* backend = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelBackend(model, &backend));
    return RetrieveFrom(backend);
  }

  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_ModelInstance* instance)
  {
    TRITONBACKEND_Model* model = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelInstanceModel(instance, &model));
    return RetrieveFrom(model);
  }

  bool enable_memory_tracker_{false};
  int default_max_batch_size_{0};
  // Shared by every instance with 'stage_metrics' set; null when the server
  // has metrics disabled.
  TRITONSERVER_MetricFamily* stage_duration_family_{nullptr};
  TRITONSERVER_MetricFamily* batch_requests_family_{nullptr};
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load an ONNX model using 'artifact_name' as the name for the ONNX
  // file/directory. If 'instance_group_kind' is not
  // TRITONSERVER_INSTANCEGROUPKIND_AUTO then use it and
  // 'instance_group_device_id' to initialize the appropriate
  // execution providers. Return in 'model_path' the full path to the
  // onnx file, return in 'session' and 'allocator' the ORT session
  // and allocator.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id, std::string* model_path,
      OrtSession** session, OrtAllocator** default_allocator,
      cudaStream_t stream);

  const std::map<std::string, std::pair<int64_t, int64_t>>& ModelOutputs()
  {
    return model_outputs_;
  }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteMaxBatch(
      const OnnxTensorInfoMap& input_tensor_infos,
      const OnnxTensorInfoMap& output_tensor_infos);
  TRITONSERVER_Error* AutoCompleteIO(
      const char* key, const OnnxTensorInfoMap& io_infos);

  // Session options used when creating a ORT session.
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> session_options_;

  // model_outputs is a map that contains unique outputs that the model must
  // provide. In the model configuration, the output in the state configuration
  // can have intersection with the outputs section of the model. If an output
  // is specified both in the output section and state section, it indicates
  // that the backend must return the output state to the client too.
  std::map<std::string, std::pair<int64_t, int64_t>> model_outputs_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());
    RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  auto& model_outputs = (*state)->model_outputs_;

  // Parse the output states in the model configuration
  triton::common::TritonJson::Value sequence_batching;
  if ((*state)->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string output_state_name;
        RETURN_IF_ERROR(
            state.MemberAsString("output_name", &output_state_name));
        auto it = model_outputs.find(output_state_name);
        if (it == model_outputs.end()) {
          model_outputs.insert({output_state_name, std::make_pair(-1, i)});
        } else {
          it->second.second = i;
        }
      }
    }
  }

  // Parse the output names in the model configuration
  triton::common::TritonJson::Value outputs;
  RETURN_IF_ERROR((*state)->ModelConfig().MemberAsArray("output", &outputs));
  for (size_t i = 0; i < outputs.ArraySize(); i++) {
    triton::common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

    std::string output_name_str;

    RETURN_IF_ERROR(output.MemberAsString("name", &output_name_str));
    auto it = model_outputs.find(output_name_str);
    if (it == model_outputs.end()) {
      model_outputs.insert({output_name_str, {i, -1}});
    } else {
      it->second.first = i;
    }
  }


  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model, true /* allow_optional */)
{
  // Create session options that will be cloned and used for each
  // instance when creating that instance's session.
  OrtSessionOptions* soptions;
  THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->CreateSessionOptions(&soptions));
  session_options_.reset(soptions);

  GraphOptimizationLevel optimization_level =
      GraphOptimizationLevel::ORT_ENABLE_ALL;
  {
    triton::common::TritonJson::Value optimization;
    if (ModelConfig().Find("optimization", &optimization)) {
      triton::common::TritonJson::Value graph;
      if (optimization.Find("graph", &graph)) {
        int64_t graph_level = 0;
        THROW_IF_BACKEND_MODEL_ERROR(graph.MemberAsInt("level", &graph_level));
        if (graph_level == -1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
        } else if (graph_level == 1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        } else if (graph_level == 2) {
          optimization_level = GraphOptimizationLevel::ORT_DISABLE_ALL;
        }
      }
    }
  }
  THROW_IF_BACKEND_MODEL_ORT_ERROR(
      ort_api->SetSessionGraphOptimizationLevel(soptions, optimization_level));

  {
    // Controls whether you want to execute operators in your graph sequentially
    // or in parallel. Usually when the model has many branches, setting this
    // option to ExecutionMode::ORT_PARALLEL will give you better performance.
    int execution_mode = 0;
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      THROW_IF_BACKEND_MODEL_ERROR(TryParseModelStringParameter(
          params, "execution_mode", &execution_mode, 0));
    }

    // 0 and 1 are the only valid values.
    if (execution_mode != 0 && execution_mode != 1) {
      throw BackendModelException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string(
               "Invalid configuration value provided. Expected values for "
               " execution_mode are 0 or 1 but got " +
               std::to_string(execution_mode) + " .")
               .c_str())));
    } else {
      THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->SetSessionExecutionMode(
          soptions, execution_mode == 0 ? ExecutionMode::ORT_SEQUENTIAL
                                        : ExecutionMode::ORT_PARALLEL));
    }
  }

  // If global threadpool is enabled, disable per session threads.
  // If it is not enabled then try to read the configs for intra and inter
  // op num threads and set them for session.
  if (OnnxLoader::IsGlobalThreadPoolEnabled()) {
    THROW_IF_BACKEND_MODEL_ORT_ERROR(
        ort_api->DisablePerSessionThreads(soptions));
  } else {
    {
      // Sets the number of threads used to parallelize the execution within
      // nodes A value of 0 means ORT will pick a default
      int intra_op_thread_count = 0;
      triton::common::TritonJson::Value params;
      if (ModelConfig().Find("parameters", &params)) {
        THROW_IF_BACKEND_MODEL_ERROR(TryParseModelStringParameter(
            params, "intra_op_thread_count", &intra_op_thread_count, 0));
      }
      if (intra_op_thread_count > 0) {
        THROW_IF_BACKEND_MODEL_ORT_ERROR(
            ort_api->SetIntraOpNumThreads(soptions, intra_op_thread_count));
      }
    }

    {
      // Sets the number of threads used to parallelize the execution of the
      // graph (across nodes) If sequential execution is enabled this value is
      // ignored A value of 0 means ORT will pick a default
      int inter_op_thread_count = 0;
      triton::common::TritonJson::Value params;
      if (ModelConfig().Find("parameters", &params)) {
        THROW_IF_BACKEND_MODEL_ERROR(TryParseModelStringParameter(
            params, "inter_op_thread_count", &inter_op_thread_count, 0));
      }
      if (inter_op_thread_count > 0) {
        THROW_IF_BACKEND_MODEL_ORT_ERROR(
            ort_api->SetInterOpNumThreads(soptions, inter_op_thread_count));
      }
    }
  }

  // Enable/disable use_device_allocator_for_initializers
  {
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      triton::common::TritonJson::Value json_value;
      const char* use_device_allocator_for_initializers_key =
          "session.use_device_allocator_for_initializers";
      if (params.Find(use_device_allocator_for_initializers_key, &json_value)) {
        std::string string_value;
        THROW_IF_BACKEND_MODEL_ERROR(
            json_value.MemberAsString("string_value", &string_value));

        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("Configuring '") +
             use_device_allocator_for_initializers_key + "' to '" +
             string_value + "' for '" + Name() + "'")
                .c_str());
        THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->AddSessionConfigEntry(
            soptions, use_device_allocator_for_initializers_key,
            string_value.c_str()));
      }
    }
  }

  // memory configs
  // enable/disable mem arena
  {
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      triton::common::TritonJson::Value json_value;
      if (params.Find("enable_mem_arena", &json_value)) {
        std::string string_value;
        THROW_IF_BACKEND_MODEL_ERROR(
            json_value.MemberAsString("string_value", &string_value));
        bool enable_cpu_mem_arena = false;
        THROW_IF_BACKEND_MODEL_ERROR(
            ParseBoolValue(string_value, &enable_cpu_mem_arena));

        OrtStatus* ort_status = nullptr;
        if (enable_cpu_mem_arena) {
          ort_status = ort_api->EnableCpuMemArena(soptions);
        } else {
          ort_status = ort_api->DisableCpuMemArena(soptions);
        }
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("Configuring enable_mem_arena to ") + string_value)
                .c_str());
        THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_status);
      }
    }
  }

  // enable/disable mem pattern
  {
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      triton::common::TritonJson::Value json_value;
      if (params.Find("enable_mem_pattern", &json_value)) {
        std::string string_value;
        THROW_IF_BACKEND_MODEL_ERROR(
            json_value.MemberAsString("string_value", &string_value));
        bool enable_mem_pattern = false;
        auto err = ParseBoolValue(string_value, &enable_mem_pattern);
        THROW_IF_BACKEND_MODEL_ERROR(err);

        OrtStatus* ort_status = nullptr;
        if (enable_mem_pattern) {
          ort_status = ort_api->EnableMemPattern(soptions);
        } else {
          ort_status = ort_api->DisableMemPattern(soptions);
        }
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("Configuring enable_mem_pattern to ") + string_value)
                .c_str());
        THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_status);
      }
    }
  }

  // FIXME. Is it possible to share a single OrtSession across
  // multiple instances? If so then should move loading and validation
  // of the session to here instead of creating a session for each
  // instance in ModelStateInstance::Create().
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name,
    const TRITONSERVER_InstanceGroupKind instance_group_kind,
    const int32_t instance_group_device_id, std::string* model_path,
    OrtSession** session, OrtAllocator** default_allocator, cudaStream_t stream)
{
  // Find the ONNX file that describes the model itself. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.onnx").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.onnx";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  // If the model path is a directory then the actual model is
  // <dir>/model.onnx.
  {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(*model_path, &is_dir));
    if (is_dir) {
      *model_path = JoinPath({*model_path, "model.onnx"});
    }
  }

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  // Make a clone for the session options for this instance...
  OrtSessionOptions* soptions;
  RETURN_IF_ORT_ERROR(
      ort_api->CloneSessionOptions(session_options_.get(), &soptions));
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> soptions_wrapper(
      soptions);

  bool need_lock = false;

  // Add execution providers if they are requested.
  // Don't need to ensure uniqueness of the providers, ONNX Runtime
  // will check it.

  // GPU execution providers
#if defined(TRITON_ENABLE_GPU) || defined(TRITON_ENABLE_ROCM)
  if ((instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
      (instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_AUTO)) {
    std::map<std::string, std::string> cuda_options_map;
    triton::common::TritonJson::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      triton::common::TritonJson::Value eas;
      if (optimization.Find("execution_accelerators", &eas)) {
        triton::common::TritonJson::Value gpu_eas;
        if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < gpu_eas.ArraySize(); ea_idx++) {
            triton::common::TritonJson::Value ea;
            RETURN_IF_ERROR(gpu_eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));
#ifdef TRITON_ENABLE_ONNXRUNTIME_TENSORRT
            if (name == kTensorRTExecutionAccelerator) {
              // create tensorrt options with default values
              OrtTensorRTProviderOptionsV2* trt_options;
              THROW_IF_BACKEND_MODEL_ORT_ERROR(
                  ort_api->CreateTensorRTProviderOptions(&trt_options));
              std::unique_ptr<
                  OrtTensorRTProviderOptionsV2,
                  decltype(ort_api->ReleaseTensorRTProviderOptions)>
                  rel_trt_options(
                      trt_options, ort_api->ReleaseTensorRTProviderOptions);
              std::string int8_calibration_table_name;
              std::string trt_engine_cache_path;
              // Validate and set parameters
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys, keys, values;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string, key, value;
                  if (param_key == "precision_mode") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "FP16") {
                      key = "trt_fp16_enable";
                      value = "1";
                    } else if (value_string == "INT8") {
                      key = "trt_int8_enable";
                      value = "1";
                    } else if (value_string != "FP32") {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("unsupported precision mode '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "max_workspace_size_bytes") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    size_t max_workspace_size_bytes;
                    RETURN_IF_ERROR(ParseUnsignedLongLongValue(
                        value_string, &max_workspace_size_bytes));
                    key = "trt_max_workspace_size";
                    value = value_string;
                  } else if (param_key == "trt_max_partition_iterations") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_max_partition_iterations;
                    RETURN_IF_ERROR(ParseIntValue(
                        value_string, &trt_max_partition_iterations));
                    key = "trt_max_partition_iterations";
                    value = value_string;
                  } else if (param_key == "trt_min_subgraph_size") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_min_subgraph_size;
                    RETURN_IF_ERROR(
                        ParseIntValue(value_string, &trt_min_subgraph_size));
                    key = "trt_min_subgraph_size";
                    value = value_string;
                  } else if (param_key == "int8_calibration_table_name") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_int8_calibration_table_name";
                  } else if (param_key == "int8_use_native_calibration_table") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool use_native_calibration_table;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &use_native_calibration_table));
                    key = "trt_int8_use_native_calibration_table";
                    value = value_string;
                  } else if (param_key == "trt_dla_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_dla_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_dla_enable));
                    key = "trt_dla_enable";
                    value = value_string;
                  } else if (param_key == "trt_dla_core") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_dla_core;
                    RETURN_IF_ERROR(ParseIntValue(value_string, &trt_dla_core));
                    key = "trt_dla_core";
                    value = value_string;
                  } else if (param_key == "trt_engine_cache_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool enable_cache;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &enable_cache));
                    key = "trt_engine_cache_enable";
                    value = value_string;
                  } else if (param_key == "trt_engine_cache_path") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_engine_cache_path";
                  } else if (param_key == "trt_engine_cache_prefix") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_engine_cache_prefix";
                  } else if (param_key == "trt_dump_subgraphs") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool dump_subgraphs;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &dump_subgraphs));
                    key = "trt_dump_subgraphs";
                    value = value_string;
                  } else if (param_key == "trt_force_sequential_engine_build") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_force_sequential_engine_build;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_force_sequential_engine_build));
                    key = "trt_force_sequential_engine_build";
                    value = value_string;
                  } else if (param_key == "trt_context_memory_sharing_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_context_memory_sharing_enable;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_context_memory_sharing_enable));
                    key = "trt_context_memory_sharing_enable";
                    value = value_string;
                  } else if (param_key == "trt_layer_norm_fp32_fallback") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_layer_norm_fp32_fallback;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_layer_norm_fp32_fallback));
                    key = "trt_layer_norm_fp32_fallback";
                    value = value_string;
                  } else if (param_key == "trt_timing_cache_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_timing_cache_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_timing_cache_enable));
                    key = "trt_timing_cache_enable";
                    value = value_string;
                  } else if (param_key == "trt_timing_cache_path") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_timing_cache_path";
                  } else if (param_key == "trt_force_timing_cache") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_force_timing_cache;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_force_timing_cache));
                    key = "trt_force_timing_cache";
                    value = value_string;
                  } else if (param_key == "trt_detailed_build_log") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_detailed_build_log;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_detailed_build_log));
                    key = "trt_detailed_build_log";
                    value = value_string;
                  } else if (param_key == "trt_build_heuristics_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_build_heuristics_enable;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_build_heuristics_enable));
                    key = "trt_build_heuristics_enable";
                    value = value_string;
                  } else if (param_key == "trt_sparsity_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_sparsity_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_sparsity_enable));
                    key = "trt_sparsity_enable";
                    value = value_string;
                  } else if (param_key == "trt_builder_optimization_level") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_builder_optimization_level;
                    RETURN_IF_ERROR(ParseIntValue(
                        value_string, &trt_builder_optimization_level));
                    key = "trt_builder_optimization_level";
                    value = value_string;
                  } else if (param_key == "trt_auxiliary_streams") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_auxiliary_streams;
                    RETURN_IF_ERROR(
                        ParseIntValue(value_string, &trt_auxiliary_streams));
                    key = "trt_auxiliary_streams";
                    value = value_string;
                  } else if (param_key == "trt_tactic_sources") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_tactic_sources";
                  } else if (param_key == "trt_extra_plugin_lib_paths") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_extra_plugin_lib_paths";
                  } else if (param_key == "trt_profile_min_shapes") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_profile_min_shapes";
                  } else if (param_key == "trt_profile_max_shapes") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_profile_max_shapes";
                  } else if (param_key == "trt_profile_opt_shapes") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_profile_opt_shapes";
                  } else if (param_key == "trt_cuda_graph_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_cuda_graph_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_cuda_graph_enable));
                    key = "trt_cuda_graph_enable";
                    value = value_string;
                  } else if (param_key == "trt_dump_ep_context_model") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_dump_ep_context_model;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_dump_ep_context_model));
                    key = "trt_dump_ep_context_model";
                    value = value_string;
                  } else if (param_key == "trt_ep_context_file_path") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_ep_context_file_path";
                  } else if (param_key == "trt_ep_context_embed_mode") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_ep_context_embed_mode;
                    RETURN_IF_ERROR(ParseIntValue(
                        value_string, &trt_ep_context_embed_mode));
                    key = "trt_ep_context_embed_mode";
                    value = value_string;
                  } else {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string(
                            "unknown parameter '" + param_key +
                            "' is provided for TensorRT Execution "
                            "Accelerator")
                            .c_str());
                  }
                  if (!key.empty() && !value.empty()) {
                    keys.push_back(key);
                    values.push_back(value);
                  }
                }

                // assign correct GPU to EP
                keys.push_back(std::string("device_id"));
                values.push_back(std::to_string(instance_group_device_id));

                std::vector<const char*> c_keys, c_values;
                if (!keys.empty() && !values.empty()) {
                  for (size_t i = 0; i < keys.size(); ++i) {
                    c_keys.push_back(keys[i].c_str());
                    c_values.push_back(values[i].c_str());
                  }
                  RETURN_IF_ORT_ERROR(ort_api->UpdateTensorRTProviderOptions(
                      rel_trt_options.get(), c_keys.data(), c_values.data(),
                      keys.size()));
                }
              }

              RETURN_IF_ORT_ERROR(
                  ort_api->SessionOptionsAppendExecutionProvider_TensorRT_V2(
                      static_cast<OrtSessionOptions*>(soptions),
                      rel_trt_options.get()));
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string("TensorRT Execution Accelerator is set for '") +
                   Name() + "' on device " +
                   std::to_string(instance_group_device_id))
                      .c_str());
              continue;
            }
#endif  // TRITON_ENABLE_ONNXRUNTIME_TENSORRT
#ifdef TRITON_ENABLE_ONNXRUNTIME_MIGRAPHX
            if (name == kMIGraphXExecutionAccelerator) {
              // MIGraphX is served by the out-of-tree plugin EP
              // (onnxruntime-ep-amdgpu -> libmigraphx-ep.so), registered once
              // against the process-wide OrtEnv in OnnxLoader::Init(). Provider
              // options are passed as string key/value pairs and the provider is
              // appended via SessionOptionsAppendExecutionProvider_V2 against the
              // plugin's OrtEpDevice for the target GPU. This instance's HIP
              // stream is passed as "user_compute_stream" (see below) so the
              // plugin adopts it instead of creating its own -- unifying Triton's
              // I/O copies and the model's compute onto one stream, matching the
              // classic built-in MIGraphXExecutionProvider. Option keys use the
              // plugin's native names; the plugin also accepts the classic
              // "migraphx_*" aliases for backward compatibility.
              std::vector<std::string> keys, values;
              std::string int8_calibration_table_name;
              std::string model_cache_dir;

              keys.push_back("device_id");
              values.push_back(std::to_string(instance_group_device_id));

              // Validate and set parameters
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string;
                  if (param_key == "precision_mode") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "FP16") {
                      keys.push_back("fp16_enable");
                      values.push_back("1");
                    } else if (value_string == "BF16") {
                      keys.push_back("bf16_enable");
                      values.push_back("1");
                    } else if (value_string == "FP8") {
                      keys.push_back("fp8_enable");
                      values.push_back("1");
                    } else if (value_string == "INT8") {
                      keys.push_back("int8_enable");
                      values.push_back("1");
                    } else if (value_string != "FP32") {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("unsupported precision mode '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "int8_calibration_table_name") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &int8_calibration_table_name));
                    keys.push_back("int8_calibration_table_name");
                    values.push_back(int8_calibration_table_name);
                  } else if (param_key == "int8_use_native_calibration_table") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int use_native_calibration_table;
                    RETURN_IF_ERROR(ParseIntValue(
                        value_string, &use_native_calibration_table));
                    keys.push_back("int8_use_native_calibration_table");
                    values.push_back(std::to_string(
                        use_native_calibration_table));
                  } else if (
                      param_key == "migraphx_model_cache_dir" ||
                      param_key == "cache_dir") {
                    // The plugin EP's cache option is "cache_dir" (the classic
                    // "migraphx_model_cache_dir" has no plugin alias, so map it
                    // here for config.pbtxt backward compatibility).
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &model_cache_dir));
                    keys.push_back("cache_dir");
                    values.push_back(model_cache_dir);
                  } else if (
                      param_key == "migraphx_max_dynamic_batch" ||
                      param_key == "max_dynamic_batch") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    size_t max_dynamic_batch;
                    RETURN_IF_ERROR(ParseUnsignedLongLongValue(
                        value_string, &max_dynamic_batch));
                    keys.push_back("max_dynamic_batch");
                    values.push_back(std::to_string(max_dynamic_batch));
                  } else if (
                      param_key == "migraphx_exhaustive_tune" ||
                      param_key == "exhaustive_tune") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    keys.push_back("exhaustive_tune");
                    values.push_back(value_string == "true" ? "1" : "0");
                  } else if (param_key == "compile_batches") {
                    // Plugin-only knob: comma-separated list of batch sizes to
                    // pre-compile (e.g. "1,2,4,8,16,32").
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    keys.push_back("compile_batches");
                    values.push_back(value_string);
                  } else if (param_key == "hip_graph_enable") {
                    // Plugin-only knob: enable hipGraph capture/replay.
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    keys.push_back("hip_graph_enable");
                    values.push_back(value_string == "true" ? "1" : "0");
                  } else if (param_key == "coalesce_io") {
                    // Plugin-only knob: coalesce I/O staging buffers.
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    keys.push_back("coalesce_io");
                    values.push_back(value_string == "true" ? "1" : "0");
                  } else if (param_key == "precompile_at_load") {
                    // Plugin-only knob: compile missing MXR programs at session
                    // create (after preload). When false, compilation is deferred
                    // to the first Compute() for each missing batch bucket.
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    keys.push_back("precompile_at_load");
                    values.push_back(value_string == "true" ? "1" : "0");
                  } else {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string(
                            "unknown parameter '" + param_key +
                            "' is provided for MIGraphX Execution "
                            "Accelerator")
                            .c_str());
                  }
                }
              }

              // Select the plugin's OrtEpDevice for the target GPU. The plugin
              // exposes one OrtEpDevice per HIP device, enumerated in device-id
              // order, all reporting Ep name "MIGraphX".
              OrtEnv* ort_env = OnnxLoader::Env();
              RETURN_ERROR_IF_TRUE(
                  ort_env == nullptr, TRITONSERVER_ERROR_INTERNAL,
                  std::string("ONNX Runtime environment is not initialized; "
                              "cannot append MIGraphX plugin EP"));

              const OrtEpDevice* const* ep_devices = nullptr;
              size_t num_ep_devices = 0;
              RETURN_IF_ORT_ERROR(
                  ort_api->GetEpDevices(ort_env, &ep_devices, &num_ep_devices));

              const OrtEpDevice* selected_device = nullptr;
              int migraphx_device_idx = 0;
              for (size_t i = 0; i < num_ep_devices; ++i) {
                const char* ep_name = ort_api->EpDevice_EpName(ep_devices[i]);
                if (ep_name != nullptr && std::string(ep_name) == "MIGraphX") {
                  // Fall back to the first MIGraphX device if the exact index is
                  // not matched.
                  if (selected_device == nullptr) {
                    selected_device = ep_devices[i];
                  }
                  if (migraphx_device_idx == instance_group_device_id) {
                    selected_device = ep_devices[i];
                    break;
                  }
                  ++migraphx_device_idx;
                }
              }

              RETURN_ERROR_IF_TRUE(
                  selected_device == nullptr, TRITONSERVER_ERROR_UNAVAILABLE,
                  std::string("MIGraphX plugin EP is not available; ensure "
                              "libmigraphx-ep.so is present and registered"));

              // Hand this instance's compute stream to the plugin so it adopts
              // the stream instead of creating its own. The V2 append is
              // string-only, so the handle is passed as a decimal pointer
              // address; the plugin parses it back to the hipStream_t (matching
              // the classic built-in EP's user_compute_stream). A non-null handle
              // implies has_user_compute_stream on the plugin side.
              if (stream != nullptr) {
                keys.push_back("user_compute_stream");
                values.push_back(std::to_string(
                    reinterpret_cast<uintptr_t>(stream)));
                keys.push_back("has_user_compute_stream");
                values.push_back("1");
              }

              std::vector<const char*> c_keys, c_values;
              for (size_t i = 0; i < keys.size(); ++i) {
                c_keys.push_back(keys[i].c_str());
                c_values.push_back(values[i].c_str());
              }
              RETURN_IF_ORT_ERROR(
                  ort_api->SessionOptionsAppendExecutionProvider_V2(
                      soptions, ort_env, &selected_device, 1, c_keys.data(),
                      c_values.data(), keys.size()));
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string(
                       "MIGraphX plugin Execution Accelerator is set for '") +
                   Name() + "' on device " +
                   std::to_string(instance_group_device_id))
                      .c_str());
              continue;
            }
#endif  // TRITON_ENABLE_ONNXRUNTIME_MIGRAPHX

            if (name == "cuda") {
              // Parse CUDA EP configurations
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string, key, value;
                  // Special handling for boolean values
                  if (param_key == "do_copy_in_default_stream" ||
                      param_key == "use_ep_level_unified_stream") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool bool_value;
                    RETURN_IF_ERROR(ParseBoolValue(value_string, &bool_value));
                    key = param_key;
                    value = value_string;
                  } else {
                    key = param_key;
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                  }
                  if (!key.empty() && !value.empty()) {
                    cuda_options_map[key] = value;
                  }
                }
              }
            } else {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") + name +
                   "' is requested")
                      .c_str());
            }
          }
        }
      }
    }

#ifdef TRITON_ENABLE_GPU
    // Default GPU execution provider.
    // Using default values for everything other than device id and cuda
    // stream
    OrtCUDAProviderOptionsV2* cuda_options;
    RETURN_IF_ORT_ERROR(ort_api->CreateCUDAProviderOptions(&cuda_options));
    std::unique_ptr<
        OrtCUDAProviderOptionsV2, decltype(ort_api->ReleaseCUDAProviderOptions)>
        rel_cuda_options(cuda_options, ort_api->ReleaseCUDAProviderOptions);
    cuda_options_map["device_id"] = std::to_string(instance_group_device_id);
    cuda_options_map["has_user_compute_stream"] = stream != nullptr ? "1" : "0";
    RETURN_IF_ORT_ERROR(ort_api->UpdateCUDAProviderOptionsWithValue(
        rel_cuda_options.get(), "default_memory_arena_cfg", nullptr));
    {
      // Parse CUDA EP configurations directly from the parameters field.
      // This is deprecated with adding support for CUDA EP in the
      // gpu_execution_accelerator field. Keeping this for backward
      // compatibility.
      triton::common::TritonJson::Value params;
      if (model_config_.Find("parameters", &params)) {
        triton::common::TritonJson::Value json_value;
        if (params.Find("cudnn_conv_algo_search", &json_value)) {
          int cudnn_conv_algo_search = 0;
          RETURN_IF_ERROR(TryParseModelStringParameter(
              params, "cudnn_conv_algo_search", &cudnn_conv_algo_search, 0));
          std::string string_value;
          switch (cudnn_conv_algo_search) {
            case 0:
              string_value = "EXHAUSTIVE";
              break;
            case 1:
              string_value = "HEURISTIC";
              break;
            case 2:
              string_value = "DEFAULT";
              break;
            default:
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unsupported cudnn_conv_algo_search value '") +
                   std::to_string(cudnn_conv_algo_search) + "' is requested")
                      .c_str());
          }
          cuda_options_map["cudnn_conv_algo_search"] = string_value;
        } else {
          cuda_options_map["cudnn_conv_algo_search"] = "EXHAUSTIVE";
        }

        if (params.Find("gpu_mem_limit", &json_value)) {
          std::string string_value;
          RETURN_IF_ERROR(
              json_value.MemberAsString("string_value", &string_value));
          cuda_options_map["gpu_mem_limit"] = string_value;
        } else {
          cuda_options_map["gpu_mem_limit"] =
              std::to_string(std::numeric_limits<size_t>::max());
        }

        if (params.Find("arena_extend_strategy", &json_value)) {
          int arena_extend_strategy = 0;
          RETURN_IF_ERROR(TryParseModelStringParameter(
              params, "arena_extend_strategy", &arena_extend_strategy, 0));
          std::string string_value;
          switch (arena_extend_strategy) {
            case 0:
              string_value = "kNextPowerOfTwo";
              break;
            case 1:
              string_value = "kSameAsRequested";
              break;
            default:
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unsupported arena_extend_strategy value '") +
                   std::to_string(arena_extend_strategy) + "' is requested")
                      .c_str());
          }
          cuda_options_map["arena_extend_strategy"] = string_value;
        } else {
          cuda_options_map["arena_extend_strategy"] = "kNextPowerOfTwo";
        }

        if (params.Find("do_copy_in_default_stream", &json_value)) {
          std::string string_value;
          RETURN_IF_ERROR(
              json_value.MemberAsString("string_value", &string_value));
          cuda_options_map["do_copy_in_default_stream"] = string_value;
        } else {
          cuda_options_map["do_copy_in_default_stream"] = "1";
        }
      }
    }

    std::vector<const char*> option_names, option_values;
    for (const auto& [key, value] : cuda_options_map) {
      option_names.push_back(key.c_str());
      option_values.push_back(value.c_str());
    }

    RETURN_IF_ORT_ERROR(ort_api->UpdateCUDAProviderOptions(
        rel_cuda_options.get(), option_names.data(), option_values.data(),
        option_values.size()));

    if (stream != nullptr) {
      RETURN_IF_ORT_ERROR(ort_api->UpdateCUDAProviderOptionsWithValue(
          rel_cuda_options.get(), "user_compute_stream", stream));
    }
    RETURN_IF_ORT_ERROR(ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2(
        soptions, cuda_options));

    OrtAllocator* allocator;
    char* options;
    RETURN_IF_ORT_ERROR(ort_api->GetAllocatorWithDefaultOptions(&allocator));
    RETURN_IF_ORT_ERROR(ort_api->GetCUDAProviderOptionsAsString(
        rel_cuda_options.get(), allocator, &options));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("CUDA Execution Accelerator is set for '") + Name() +
         "' on device " + std::to_string(instance_group_device_id) +
         std::string(" with options: ") + std::string(options))
            .c_str());
    RETURN_IF_ORT_ERROR(ort_api->AllocatorFree(allocator, options));
#endif  // TRITON_ENABLE_GPU

#if defined(TRITON_ENABLE_ROCM) && !defined(TRITON_ENABLE_ONNXRUNTIME_MIGRAPHX)
    // Default AMD GPU execution provider using ROCm
    // Using default values for everything other than device id and ROCM
    // stream
    OrtROCMProviderOptions rocm_options;
    rocm_options.device_id = instance_group_device_id;
    rocm_options.has_user_compute_stream = stream != nullptr ? 1 : 0;
    rocm_options.user_compute_stream =
        stream != nullptr ? (void*)stream : nullptr,
    rocm_options.default_memory_arena_cfg = nullptr;

    {
      // Parse ROCm EP configurations
      triton::common::TritonJson::Value params;
      if (model_config_.Find("parameters", &params)) {
        int miopen_conv_exhaustive_search = 0;
        RETURN_IF_ERROR(TryParseModelStringParameter(
            params, "miopen_conv_exhaustive_search", &miopen_conv_exhaustive_search, 0));
        rocm_options.miopen_conv_exhaustive_search = miopen_conv_exhaustive_search;

        RETURN_IF_ERROR(TryParseModelStringParameter(
            params, "gpu_mem_limit", &rocm_options.gpu_mem_limit,
            std::numeric_limits<size_t>::max()));

        RETURN_IF_ERROR(TryParseModelStringParameter(
            params, "arena_extend_strategy",
            &rocm_options.arena_extend_strategy, 0));

        RETURN_IF_ERROR(TryParseModelStringParameter(
            params, "do_copy_in_default_stream",
            &rocm_options.do_copy_in_default_stream, true));

        RETURN_IF_ERROR(TryParseModelStringParameter(
            params, "tunable_op_enable",
            &rocm_options.tunable_op_enable, true));

        RETURN_IF_ERROR(TryParseModelStringParameter(
            params, "tunable_op_tuning_enable",
            &rocm_options.tunable_op_tuning_enable, true));

        int tunable_op_max_tuning_ms = 0;
        RETURN_IF_ERROR(TryParseModelStringParameter(
            params, "tunable_op_max_tuning_duration_ms", &tunable_op_max_tuning_ms, 0));
        rocm_options.tunable_op_max_tuning_duration_ms = tunable_op_max_tuning_ms;
      }
    }

    RETURN_IF_ORT_ERROR(ort_api->SessionOptionsAppendExecutionProvider_ROCM(
        soptions, &rocm_options));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("ROCM Execution Accelerator is set for '") + Name() +
         "' on device " + std::to_string(instance_group_device_id))
            .c_str());
#endif  // TRITON_ENABLE_ROCM
  }

#endif  // TRITON_ENABLE_GPU || TRITON_ENABLE_ROCM

  // CPU execution providers
  {
    triton::common::TritonJson::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      triton::common::TritonJson::Value eas;
      if (optimization.Find("execution_accelerators", &eas)) {
        triton::common::TritonJson::Value cpu_eas;
        if (eas.Find("cpu_execution_accelerator", &cpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < cpu_eas.ArraySize(); ea_idx++) {
            triton::common::TritonJson::Value ea;
            RETURN_IF_ERROR(cpu_eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));
#ifdef TRITON_ENABLE_ONNXRUNTIME_OPENVINO
            if (name == kOpenVINOExecutionAccelerator) {
              need_lock = true;
              OrtOpenVINOProviderOptions openvino_options;
              openvino_options.device_type =
                  "CPU";  // device_type default is CPU

              RETURN_IF_ORT_ERROR(
                  ort_api->SessionOptionsAppendExecutionProvider_OpenVINO(
                      soptions, &openvino_options));

              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string("OpenVINO Execution Accelerator is set for '") +
                   Name() + "' on CPU")
                      .c_str());
              continue;
            }
#endif  // TRITON_ENABLE_ONNXRUNTIME_OPENVINO
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("unknown Execution Accelerator '") + name +
                 "' is requested")
                    .c_str());
          }
        }
      }
    }
  }

  // Register all op libraries that contain custom operations.
  {
    triton::common::TritonJson::Value model_ops;
    if (model_config_.Find("model_operations", &model_ops)) {
      triton::common::TritonJson::Value op_library_filenames;
      if (model_ops.Find("op_library_filename", &op_library_filenames)) {
        for (size_t op_idx = 0; op_idx < op_library_filenames.ArraySize();
             op_idx++) {
          std::string op_filename;
          RETURN_IF_ERROR(
              op_library_filenames.IndexAsString(op_idx, &op_filename));
          void* library_handle = nullptr;
          RETURN_IF_ORT_ERROR(ort_api->RegisterCustomOpsLibrary(
              soptions, op_filename.c_str(), &library_handle));
        }
      }
    }
  }

  // ONNX session creation with OpenVINO is not thread-safe,
  // so multiple creations are serialized with a global lock.
  static std::mutex global_context_mu;
  std::unique_lock<std::mutex> glock(global_context_mu, std::defer_lock);
  if (need_lock) {
    glock.lock();
  }

  RETURN_IF_ERROR(OnnxLoader::LoadSession(
      true /* is_path */, *model_path, soptions, session));

  // get default cpu allocator
  RETURN_IF_ORT_ERROR(
      ort_api->GetAllocatorWithDefaultOptions(default_allocator));

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // If the model configuration already specifies inputs and outputs
  // then don't perform any auto-completion.
  size_t input_cnt = 0;
  size_t output_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (ModelConfig().Find("input", &inputs)) {
      input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (ModelConfig().Find("batch_input", &config_batch_inputs)) {
      input_cnt += config_batch_inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if (ModelConfig().Find("output", &outputs)) {
      output_cnt = outputs.ArraySize();
    }
  }

  if ((input_cnt > 0) && (output_cnt > 0)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("skipping model configuration auto-complete for '") +
         Name() + "': inputs and outputs already specified")
            .c_str());
    return nullptr;  // success
  }

  std::string artifact_name;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsString("default_model_filename", &artifact_name));

  // Must cleanup 'session'. 'allocator' is default allocator which
  // is managed by ONNX Runtime so don't need to free/release
  std::unique_ptr<OrtSession, SessionDeleter> session;
  OrtAllocator* default_allocator;
  std::string model_path;
  {
    TRITONSERVER_InstanceGroupKind kind = TRITONSERVER_INSTANCEGROUPKIND_CPU;

#if defined(TRITON_ENABLE_GPU) || defined(TRITON_ENABLE_ROCM)
    triton::common::TritonJson::Value instance_group;
    ModelConfig().Find("instance_group", &instance_group);

    // Earlier in the model lifecycle, device checks for the instance group
    // have already occurred. If at least one instance group with
    // "kind" = "KIND_GPU" then allow model to use GPU else autocomplete to
    // "KIND_CPU"
    for (size_t i = 0; i < instance_group.ArraySize(); ++i) {
      triton::common::TritonJson::Value instance_obj;
      RETURN_IF_ERROR(instance_group.IndexAsObject(i, &instance_obj));

      triton::common::TritonJson::Value instance_group_kind;
      instance_obj.Find("kind", &instance_group_kind);
      std::string kind_str;
      RETURN_IF_ERROR(instance_group_kind.AsString(&kind_str));

      if (kind_str == "KIND_GPU") {
        kind = TRITONSERVER_INSTANCEGROUPKIND_GPU;
        break;
      }
    }
#endif  // TRITON_ENABLE_GPU || TRITON_ENABLE_ROCM

    OrtSession* sptr = nullptr;
    RETURN_IF_ERROR(LoadModel(
        artifact_name, kind, 0, &model_path, &sptr, &default_allocator,
        nullptr));
    session.reset(sptr);
  }
  OnnxTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(
      InputInfos(session.get(), default_allocator, input_tensor_infos));
  OnnxTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(
      OutputInfos(session.get(), default_allocator, output_tensor_infos));
  RETURN_IF_ERROR(
      AutoCompleteMaxBatch(input_tensor_infos, output_tensor_infos));
  if (input_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("input", input_tensor_infos));
  }
  if (output_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("output", output_tensor_infos));
  }

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteMaxBatch(
    const OnnxTensorInfoMap& input_tensor_infos,
    const OnnxTensorInfoMap& output_tensor_infos)
{
  // Determine if the model can potentially support batching. All
  // input and output tensors must have a variable first dimension.
  bool can_support_batching = true;
  for (const auto& io_info : input_tensor_infos) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      can_support_batching = false;
    }
  }
  for (const auto& io_info : output_tensor_infos) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      can_support_batching = false;
    }
  }

  // Set max-batch-size to 1 if we have determined that batching is
  // supported and max-batch-size is not specified. We need to update
  // the configuration itself as well as the cached value we have already
  // initialized in the model state.
  if (can_support_batching) {
    if (MaxBatchSize() == 0) {
      int default_max_batch_size = 0;
      {
        TRITONBACKEND_Backend* backend;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_ModelBackend(TritonModel(), &backend));
        void* state;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_BackendState(backend, &state));
        default_max_batch_size = reinterpret_cast<BackendConfiguration*>(state)
                                     ->default_max_batch_size_;
      }
      int max_batch_size = std::max(default_max_batch_size, 0);

      triton::common::TritonJson::Value mbs_value;
      ModelConfig().Find("max_batch_size", &mbs_value);
      mbs_value.SetInt(max_batch_size);
      SetMaxBatchSize(max_batch_size);

      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string(
               "autofilled max_batch_size to " +
               std::to_string(max_batch_size) + " for model '") +
           Name() +
           "' since batching is supporrted but no max_batch_size is "
           "specified "
           "in model configuration. Must specify max_batch_size to utilize "
           "autofill with a larger max batch size")
              .c_str());
    }

    // Check to see if we need to turn on dynamic batching
    // since model supports batching
    if (MaxBatchSize() > 1) {
      triton::common::TritonJson::Value value;
      bool found_sequence_batching =
          ModelConfig().Find("sequence_batching", &value);
      bool found_dynamic_batching =
          ModelConfig().Find("dynamic_batching", &value);
      if (!found_sequence_batching && !found_dynamic_batching) {
        triton::common::TritonJson::Value dynamic_batching(
            ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_ERROR(
            ModelConfig().Add("dynamic_batching", std::move(dynamic_batching)));
      }
    }

  } else if (MaxBatchSize() != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("autofill failed for model '") + Name() +
         "': model does not support batching while non-zero max_batch_size"
         " is specified")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteIO(const char* key, const OnnxTensorInfoMap& io_infos)
{
  triton::common::TritonJson::Value existing_ios;
  bool found_ios = ModelConfig().Find(key, &existing_ios);

  triton::common::TritonJson::Value ios(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io_info : io_infos) {
    triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERROR(io.AddString("name", io_info.first));
    RETURN_IF_ERROR(io.AddString(
        "data_type", OnnxDataTypeToModelConfigDataType(io_info.second.type_)));

    // The model signature supports batching then the first dimension
    // is -1 and should not appear in the model configuration 'dims'
    // that we are creating.
    const auto& io_info_dims = io_info.second.dims_;
    triton::common::TritonJson::Value dims(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
    for (size_t i = (MaxBatchSize() > 0) ? 1 : 0; i < io_info_dims.size();
         ++i) {
      RETURN_IF_ERROR(dims.AppendInt(io_info_dims[i]));
    }

    // If dims are empty then must use a reshape...
    if (dims.ArraySize() == 0) {
      RETURN_IF_ERROR(dims.AppendInt(1));
      triton::common::TritonJson::Value reshape(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      triton::common::TritonJson::Value reshape_dims(
          ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
      RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
      // Empty reshape with `max_batch_size` indicates a scalar tensor in the
      // model configuration which is not a valid model configuration.
      if (MaxBatchSize() > 0) {
        RETURN_IF_ERROR(io.Add("reshape", std::move(reshape)));
      }
    }
    RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
    RETURN_IF_ERROR(ios.Append(std::move(io)));
  }

  if (found_ios) {
    existing_ios.Swap(ios);
  } else {
    ModelConfig().Add(key, std::move(ios));
  }

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  void ReleaseOrtRunResources(OrtIoBinding* binding);
  void SynchronizeStream();
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
  TRITONSERVER_Error* OrtRun(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count, OrtIoBinding* binding);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, bool* cuda_copy, size_t slot,
      OrtIoBinding** binding);
  TRITONSERVER_Error* InitPersistentInputBinding();
  TRITONSERVER_Error* SetInputTensorsPersistent(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count, BackendInputCollector* collector,
      bool* cuda_copy, size_t slot, OrtIoBinding** binding);
  bool GatherPersistentInputsDirect(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      size_t total_batch_size, size_t slot);
  TRITONSERVER_Error* BuildOutputPlan(TRITONBACKEND_Request* request);
  TRITONSERVER_Error* SetStringInputTensor(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
      std::vector<const char*>* string_ptrs, bool* cuda_copy);
  void SetStringInputBuffer(
      const std::string& name, const std::vector<size_t>& expected_byte_sizes,
      const std::vector<size_t>& expected_element_cnts,
      std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
      std::vector<const char*>* string_ptrs);
  void FillStringData(std::vector<const char*>* string_ptrs, size_t cnt);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, OrtIoBinding* binding,
      size_t slot);

  TRITONSERVER_Error* ReadOutputTensor(
      std::vector<int64_t>& batchn_shape, TRITONSERVER_DataType& dtype,
      OrtValue* output_tensor, void** output_buffer,
      std::vector<std::vector<char>>& string_buffers,
      std::vector<size_t>& offsets);
  bool SetStringOutputBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool SetStringStateBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool SetStringBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, bool state);

  // Per-request statistics and release for one execution. Triton's request
  // array is only valid during Execute, so the request pointers are copied.
  struct Completion {
    std::vector<TRITONBACKEND_Request*> requests;
    std::vector<uint8_t> success;
    size_t total_batch_size{0};
    bool batch_ok{false};
    uint64_t exec_start_ns{0};
    uint64_t compute_start_ns{0};
    uint64_t compute_end_ns{0};
    uint64_t exec_end_ns{0};
  };
  // An execution's responses plus the completion that must follow them.
  struct ResponseBatch {
    std::vector<TRITONBACKEND_Response*> responses;
    Completion completion;
  };
  TRITONSERVER_Error* ReadThreadingParameters();
  void StartBackgroundThreads();
  void CompleteExecution(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>&& responses,
      size_t total_batch_size, bool all_response_failed,
      uint64_t exec_start_ns, uint64_t compute_start_ns,
      uint64_t compute_end_ns, uint64_t exec_end_ns);
  void SendAndComplete(ResponseBatch& batch);
  void DispatchCompletion(Completion&& completion);
  void FinishCompletion(Completion& completion);

  ModelState* model_state_;

  // Background threads, all off by default. With 'completion_threads' set,
  // statistics and request release run on that many workers instead of the
  // instance thread. With 'async_response_send' set, one sender thread sends
  // the responses, in execution order, and then hands the batch on for
  // completion.
  int completion_thread_count_{0};
  int async_response_send_{0};
  WorkQueue<Completion> completion_q_;
  WorkQueue<ResponseBatch> response_q_;

  // With 'stage_metrics' set, every successful execution observes its stage
  // durations and request count into histograms on the metrics endpoint.
  // Runner wait is only observed when pipelined.
  enum Stage { kGather, kRunnerWait, kRun, kOutput, kStageCount };
  int stage_metrics_{0};
  std::array<TRITONSERVER_Metric*, kStageCount> stage_duration_metrics_{};
  TRITONSERVER_Metric* batch_requests_metric_{nullptr};
  void InitStageMetrics();
  void ReleaseStageMetrics();
  void ObserveStages(
      uint64_t exec_start_ns, uint64_t gather_end_ns,
      uint64_t compute_start_ns, uint64_t compute_end_ns,
      uint64_t exec_end_ns, uint32_t request_count);

  // The full path to the ONNX model file.
  std::string model_path_;

  // Onnx Runtime variables that are used across runs on this
  // instance.
  OrtSession* session_;
  OrtAllocator* default_allocator_;
  OrtMemoryInfo* cuda_allocator_info_;
  const OrtMemoryInfo* cpu_allocator_info_;
  // Binding for models that don't use persistent input binding. The binding
  // for a run is passed explicitly: either this one or a
  // 'persistent_bindings_' entry.
  OrtIoBinding* transient_binding_{nullptr};
  OrtRunOptions* runOptions_;
  // map of output name -> tensor info
  OnnxTensorInfoMap output_tensor_infos_;

  // map of input name -> tensor info
  OnnxTensorInfoMap input_tensor_infos_;

  // A map from scalar output tensors to the dimension specified in model config
  std::unordered_map<std::string, std::vector<int64_t>> scalar_outputs_;

  // Persistent input binding. Eligible models gather into instance-owned
  // buffers and keep their OrtValues and IoBinding alive across runs, so the
  // steady-state cost of an input is one collector gather and nothing else.
  // Everything the per-run loop needs for one input, resolved at load and
  // never modified afterwards, so 'name.c_str()' stays valid for ORT.
  // Each staging slot is a separate pinned buffer, so one batch can be
  // gathered while another is still being read by the GPU.
  struct StagingSlot {
    BackendMemory* memory{nullptr};
    char* base{nullptr};
    TRITONSERVER_MemoryType memory_type{TRITONSERVER_MEMORY_CPU};
    int64_t memory_type_id{0};
  };
  struct PersistentInput {
    std::string name;
    std::vector<int64_t> dims;
    ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    size_t row_bytes{0};
    std::vector<StagingSlot> slots;
  };

  // The OrtValue shape bakes in the batch dimension, so a binding is only
  // reusable for the batch size it was built at.
  struct PersistentBinding {
    OrtIoBinding* binding{nullptr};
    std::vector<OrtValue*> input_tensors;
    std::vector<OrtValue*> output_tensors;
  };

  bool use_persistent_input_binding_{false};
  std::vector<PersistentInput> persistent_inputs_;

  // With 'persistent_output_binding' set, outputs also get slot-owned
  // buffers bound once per slot and batch size, in the same order as
  // 'output_plan_', so a run allocates, binds and releases no outputs.
  // Needs persistent inputs and every output batched with fixed dims.
  int persistent_output_binding_{0};
  bool use_persistent_output_binding_{false};
  std::vector<PersistentInput> persistent_outputs_;
  TRITONSERVER_Error* InitPersistentOutputs(size_t max_batch);
  TRITONSERVER_Error* AllocateStagingSlots(
      PersistentInput* tensor, size_t max_batch);
  // Indexed [slot][batch size], index 0 unused. Every entry is built at
  // load and never modified afterwards; each slot's OrtValues wrap that
  // slot's own buffers.
  size_t staging_slots_{1};
  std::vector<std::vector<PersistentBinding>> persistent_bindings_;
  TRITONSERVER_Error* BuildPersistentBinding(
      size_t slot, size_t batch_size, PersistentBinding* out);
  void ReleasePersistentBinding(PersistentBinding* pb);
  void ReleasePersistentState();

  // With 'pipeline_depth' set, Execute gathers into a free staging slot,
  // binds outputs and hands the batch to a runner thread, so the next batch
  // can be gathered while this one runs. There are pipeline_depth + 1 slots.
  // The runner alone touches ORT output state for a pipelined batch.
  struct RunJob {
    size_t slot{0};
    OrtIoBinding* binding{nullptr};
    size_t total_batch_size{0};
    bool all_response_failed{false};
    uint64_t exec_start_ns{0};
    uint64_t gather_end_ns{0};
    std::vector<TRITONBACKEND_Request*> requests;
    std::vector<TRITONBACKEND_Response*> responses;
  };
  size_t AcquireSlot();
  void WaitForFreeSlot();
  void ReleaseSlot(size_t slot);
  void RunPipelined(RunJob& job);
  int pipeline_depth_{0};
  bool pipeline_{false};
  std::mutex slot_mu_;
  std::condition_variable slot_cv_;
  std::vector<size_t> free_slots_;
  WorkQueue<RunJob> runner_q_;

  // With 'direct_input_gather' set, persistent inputs are copied straight
  // from the request buffers instead of through BackendInputCollector. Any
  // case the direct path can't handle sends the whole execution back to the
  // collector, which rewrites every staging row.
  int direct_input_gather_{0};
  bool direct_gather_warned_{false};
  bool GatherInputRows(
      const PersistentInput& pi, TRITONBACKEND_Request* request, char* base,
      size_t capacity, size_t* offset);

  // With 'gather_threads' > 1 the direct gather is split across threads.
  // Each part owns a fixed set of inputs, assigned at load to balance bytes
  // per row, and writes only those inputs' staging buffers and its own
  // result flag, so the parts share no mutable state.
  bool GatherPersistentInputs(size_t part);
  int gather_thread_count_{1};
  std::vector<std::vector<size_t>> gather_parts_;
  std::vector<uint8_t> gather_part_ok_;
  TRITONBACKEND_Request** gather_requests_{nullptr};
  uint32_t gather_request_count_{0};
  size_t gather_total_batch_size_{0};
  size_t gather_slot_{0};
  ForkJoin gather_fj_;

  // Per-output metadata resolved once, so the per-run loops do no name
  // hashing and no repeated device-preference queries.
  struct OutputPlan {
    const std::string* name{nullptr};
    const char* name_cstr{nullptr};
    TRITONSERVER_MemoryType memory_type{TRITONSERVER_MEMORY_CPU};
    int64_t memory_type_id{0};
    const OrtMemoryInfo* allocator_info{nullptr};
    const BatchOutput* batch_output{nullptr};
    int64_t tensor_index{-1};
    int64_t state_index{-1};
    const std::vector<int64_t>* scalar_dims{nullptr};
    // Model dims with the batch dimension first; only set when every other
    // dim is fixed, so the shape follows from the batch size alone.
    TRITONSERVER_DataType dtype{TRITONSERVER_TYPE_INVALID};
    std::vector<int64_t> fixed_dims;
  };
  std::vector<OutputPlan> output_plan_;
  bool output_plan_built_{false};
  // Owned by whichever thread reads outputs. An output uses its fixed shape
  // once ORT has confirmed it at the current batch size; any mismatch turns
  // the fixed shape off for that output.
  std::vector<uint8_t> output_fixed_ok_;
  std::vector<uint8_t> output_batch_confirmed_;

  // Reused across outputs and runs in ReadOutputTensors.
  std::vector<int64_t> output_shape_scratch_;
  std::vector<std::vector<char>> output_string_scratch_;
  std::vector<size_t> output_offsets_scratch_;

  // Persistent outputs are written straight into responses using each
  // request's batch size and requested outputs, read once per execution.
  std::vector<int64_t> request_batch_scratch_;
  std::vector<uint8_t> request_wants_scratch_;  // [request][output]
  TRITONSERVER_Error* PrepareDirectOutputs(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool WriteDirectOutput(
      size_t idx, TRITONSERVER_DataType dtype,
      std::vector<int64_t>& batchn_shape, const char* buffer,
      TRITONSERVER_MemoryType memory_type, int64_t memory_type_id,
      std::vector<TRITONBACKEND_Response*>* responses);

  // Onnx Runtime variables that will be reset and used for every run
  // on this instance.
  std::vector<OrtValue*> input_tensors_;
  std::vector<OrtValue*> output_tensors_;
  OrtValue** output_buffer_;
  std::vector<BackendMemory*> input_tensor_memories_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), session_(nullptr), default_allocator_(nullptr),
      cuda_allocator_info_(nullptr), cpu_allocator_info_(nullptr),
      output_buffer_(nullptr)
{
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), Kind(), DeviceId(), &model_path_, &session_,
      &default_allocator_, CudaStream()));

#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->CreateMemoryInfo(
        "Cuda", OrtAllocatorType::OrtArenaAllocator, DeviceId(),
        OrtMemTypeDefault, &cuda_allocator_info_));
  }
#endif //TRITON_ENABLE_GPU

#ifdef TRITON_ENABLE_ROCM
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->CreateMemoryInfo(
        "Hip", OrtAllocatorType::OrtArenaAllocator, DeviceId(),
        OrtMemTypeDefault, &cuda_allocator_info_));
  }
#endif //TRITON_ENABLE_ROCM
  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->AllocatorGetInfo(default_allocator_, &cpu_allocator_info_));

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->CreateIoBinding(session_, &transient_binding_));

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->CreateRunOptions(&runOptions_));

  // Read configs that needs to be set in RunOptions
  triton::common::TritonJson::Value params;
  if (model_state->ModelConfig().Find("parameters", &params)) {
    triton::common::TritonJson::Value json_value;
    const char* enable_memory_arena_shrinkage_key =
        "memory.enable_memory_arena_shrinkage";
    if (params.Find(enable_memory_arena_shrinkage_key, &json_value)) {
      std::string string_value;
      THROW_IF_BACKEND_MODEL_ERROR(
          json_value.MemberAsString("string_value", &string_value));

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Configuring ") + enable_memory_arena_shrinkage_key +
           " to " + string_value)
              .c_str());
      THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->AddRunConfigEntry(
          runOptions_, enable_memory_arena_shrinkage_key,
          string_value.c_str()));
    }
  }

  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
      // Skip the optional inputs which are initializers
      for (size_t i = 0; i < inputs.ArraySize(); i++) {
        triton::common::TritonJson::Value input;
        THROW_IF_BACKEND_INSTANCE_ERROR(inputs.IndexAsObject(i, &input));
        bool is_optional;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            input.MemberAsBool("optional", &is_optional));
        if (is_optional) {
          expected_input_cnt--;
        }
      }
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (model_state->ModelConfig().Find("batch_input", &config_batch_inputs)) {
      expected_input_cnt += config_batch_inputs.ArraySize();
    }
  }

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  triton::common::TritonJson::Value sequence_batching;
  if (model_state->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_START", false /* required */,
        &have_start));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_END", false /* required */,
        &have_end));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_READY", false /* required */,
        &have_ready));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateTypedSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_CORRID", false /* required */,
        &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }

    // Add the state inputs to the expected count
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      expected_input_cnt += states.ArraySize();
    }
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
  THROW_IF_BACKEND_INSTANCE_ERROR(ReadThreadingParameters());
  THROW_IF_BACKEND_INSTANCE_ERROR(InitPersistentInputBinding());
  InitStageMetrics();
  StartBackgroundThreads();
}

ModelInstanceState::~ModelInstanceState()
{
  // Workers drain their queues before exiting, so every queued request is
  // still run, sent, reported and released. Each stage feeds the next, so
  // they stop in pipeline order.
  runner_q_.Stop();
  response_q_.Stop();
  completion_q_.Stop();
  gather_fj_.Stop();
  ReleaseStageMetrics();

  ReleaseOrtRunResources(transient_binding_);
  ort_api->ReleaseRunOptions(runOptions_);
  ReleasePersistentState();
  ort_api->ReleaseIoBinding(transient_binding_);
  ort_api->ReleaseMemoryInfo(cuda_allocator_info_);
  if (session_ != nullptr) {
    OnnxLoader::UnloadSession(session_);
  }
  // 'default_allocator_' is default allocator which is managed by ONNX
  // Runtime
}

void
ModelInstanceState::ReleaseOrtRunResources(OrtIoBinding* binding)
{
  // Persistent inputs stay bound to pointer-stable staging across runs.
  if (!use_persistent_input_binding_) {
    ort_api->ClearBoundInputs(binding);
    for (auto& tensor : input_tensors_) {
      if (tensor != nullptr) {
        ort_api->ReleaseValue(tensor);
      }
    }
    input_tensors_.clear();
  }

  // Outputs are allocated by ORT on every run unless they are persistent,
  // in which case they stay bound to slot-owned buffers.
  if (!use_persistent_output_binding_) {
    ort_api->ClearBoundOutputs(binding);
    for (auto& tensor : output_tensors_) {
      if (tensor != nullptr) {
        ort_api->ReleaseValue(tensor);
      }
    }
    output_tensors_.clear();
  }

  // next release the allocated buffer using the specified allocator
  if (output_buffer_) {
    auto free_status =
        ort_api->AllocatorFree(default_allocator_, output_buffer_);
    output_buffer_ = nullptr;
    if (free_status != nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("onnx runtime allocator free error:") +
           std::to_string(ort_api->GetErrorCode(free_status)) +
           ort_api->GetErrorMessage(free_status))
              .c_str());
      ort_api->ReleaseStatus(free_status);
    }
  }

  for (BackendMemory* mem : input_tensor_memories_) {
    delete mem;
  }
  input_tensor_memories_.clear();
}

void
ModelInstanceState::SynchronizeStream()
{
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaStreamSynchronize(CudaStream());
  }
#endif  // TRITON_ENABLE_GPU
#ifdef TRITON_ENABLE_ROCM
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    static_cast<void>(
        hipStreamSynchronize(static_cast<hipStream_t>(CudaStream())));
  }
#endif  // TRITON_ENABLE_ROCM
}

TRITONSERVER_Error*
ModelInstanceState::ReadThreadingParameters()
{
  triton::common::TritonJson::Value params;
  if (model_state_->ModelConfig().Find("parameters", &params)) {
    RETURN_IF_ERROR(TryParseModelStringParameter(
        params, "completion_threads", &completion_thread_count_, 0));
    RETURN_IF_ERROR(TryParseModelStringParameter(
        params, "async_response_send", &async_response_send_, 0));
    RETURN_IF_ERROR(TryParseModelStringParameter(
        params, "direct_input_gather", &direct_input_gather_, 0));
    RETURN_IF_ERROR(TryParseModelStringParameter(
        params, "pipeline_depth", &pipeline_depth_, 0));
    RETURN_IF_ERROR(TryParseModelStringParameter(
        params, "gather_threads", &gather_thread_count_, 1));
    RETURN_IF_ERROR(TryParseModelStringParameter(
        params, "stage_metrics", &stage_metrics_, 0));
    RETURN_IF_ERROR(TryParseModelStringParameter(
        params, "persistent_output_binding", &persistent_output_binding_, 0));
  }
  RETURN_ERROR_IF_TRUE(
      completion_thread_count_ < 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("completion_threads must be >= 0"));
  RETURN_ERROR_IF_TRUE(
      gather_thread_count_ < 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("gather_threads must be >= 0"));
  RETURN_ERROR_IF_TRUE(
      pipeline_depth_ < 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("pipeline_depth must be >= 0"));
  pipeline_depth_ = std::min(pipeline_depth_, 4);
  // Read before InitPersistentInputBinding, which allocates the slots.
  staging_slots_ = static_cast<size_t>(pipeline_depth_) + 1;
  return nullptr;
}

void
ModelInstanceState::StartBackgroundThreads()
{
  if (completion_thread_count_ > 0) {
    completion_q_.Start(
        completion_thread_count_, 4 /* max_depth */,
        [this](Completion& completion) { FinishCompletion(completion); });
  }
  // Exactly one sender, so responses leave in the order the batches ran.
  if (async_response_send_ != 0) {
    response_q_.Start(1, 4 /* max_depth */, [this](ResponseBatch& batch) {
      SendAndComplete(batch);
    });
  }
  if (!use_persistent_input_binding_) {
    direct_input_gather_ = 0;
    pipeline_depth_ = 0;
  }
  if (direct_input_gather_ != 0) {
    const size_t parts = std::max<size_t>(
        1, std::min<size_t>(gather_thread_count_, persistent_inputs_.size()));
    gather_thread_count_ = static_cast<int>(parts);

    // Largest rows first, dealt round-robin, balances both the input count
    // and the bytes each part copies.
    std::vector<size_t> order(persistent_inputs_.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [this](size_t a, size_t b) {
      return persistent_inputs_[a].row_bytes > persistent_inputs_[b].row_bytes;
    });
    gather_parts_.assign(parts, {});
    for (size_t k = 0; k < order.size(); ++k) {
      gather_parts_[k % parts].push_back(order[k]);
    }
    gather_part_ok_.assign(parts, 0);
    gather_fj_.Start(parts, [this](size_t part) {
      gather_part_ok_[part] = GatherPersistentInputs(part);
    });
  } else {
    gather_thread_count_ = 1;
  }
  pipeline_ = (pipeline_depth_ > 0);
  if (pipeline_) {
    for (size_t slot = 0; slot < staging_slots_; ++slot) {
      free_slots_.push_back(slot);
    }
    runner_q_.Start(
        1, staging_slots_ /* max_depth */,
        [this](RunJob& job) { RunPipelined(job); });
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("completion threads: ") +
       std::to_string(completion_thread_count_) +
       ", async response send: " + (response_q_.Running() ? "on" : "off") +
       ", direct input gather: " + (direct_input_gather_ ? "on" : "off") +
       ", gather threads: " + std::to_string(gather_thread_count_) +
       ", pipeline depth: " + std::to_string(pipeline_depth_) +
       ", stage metrics: " + (stage_metrics_ ? "on" : "off"))
          .c_str());
}

void
ModelInstanceState::InitStageMetrics()
{
  if (stage_metrics_ == 0) {
    return;
  }
#ifndef TRITON_ENABLE_STATS
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      "stage_metrics needs a backend built with statistics; ignoring it");
  stage_metrics_ = 0;
  return;
#endif  // !TRITON_ENABLE_STATS

  const auto& config =
      BackendConfiguration::RetrieveFrom(TritonModelInstance());
  if ((config.stage_duration_family_ == nullptr) ||
      (config.batch_requests_family_ == nullptr)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "stage_metrics is set but server metrics are unavailable; ignoring it");
    stage_metrics_ = 0;
    return;
  }

  const std::string version = std::to_string(model_state_->Version());
  auto new_histogram = [&](TRITONSERVER_MetricFamily* family,
                           const char* stage,
                           const std::vector<double>& buckets,
                           TRITONSERVER_Metric** metric) {
    std::vector<std::pair<const char*, const char*>> kv{
        {"model", model_state_->Name().c_str()},
        {"version", version.c_str()},
        {"instance", Name().c_str()}};
    if (stage != nullptr) {
      kv.emplace_back("stage", stage);
    }
    std::vector<const TRITONSERVER_Parameter*> labels;
    for (const auto& [key, value] : kv) {
      labels.push_back(
          TRITONSERVER_ParameterNew(key, TRITONSERVER_PARAMETER_STRING, value));
    }
    TRITONSERVER_MetricArgs* args = nullptr;
    TRITONSERVER_Error* err = TRITONSERVER_MetricArgsNew(&args);
    if (err == nullptr) {
      err = TRITONSERVER_MetricArgsSetHistogram(
          args, buckets.data(), buckets.size());
    }
    if (err == nullptr) {
      err = TRITONSERVER_MetricNewWithArgs(
          metric, family, labels.data(), labels.size(), args);
    }
    if (args != nullptr) {
      LOG_IF_ERROR(
          TRITONSERVER_MetricArgsDelete(args), "failed to delete metric args");
    }
    for (const auto* label : labels) {
      TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
    }
    return err;
  };

  const std::vector<double> duration_buckets{
      50,   100,  200,  500,   1000,  1500,  2000,
      3000, 4000, 6000, 8000, 12000, 20000, 50000};
  std::vector<double> request_buckets;
  const int max_requests = std::max(1, model_state_->MaxBatchSize());
  for (int r = 1; r <= std::min(max_requests, 64); ++r) {
    request_buckets.push_back(r);
  }

  static const char* stage_names[kStageCount] = {
      "gather", "runner_wait", "run", "output"};
  TRITONSERVER_Error* err = nullptr;
  for (size_t s = 0; (s < kStageCount) && (err == nullptr); ++s) {
    err = new_histogram(
        config.stage_duration_family_, stage_names[s], duration_buckets,
        &stage_duration_metrics_[s]);
  }
  if (err == nullptr) {
    err = new_histogram(
        config.batch_requests_family_, nullptr, request_buckets,
        &batch_requests_metric_);
  }
  if (err != nullptr) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("failed to create stage metrics, ignoring them: ") +
         TRITONSERVER_ErrorMessage(err))
            .c_str());
    TRITONSERVER_ErrorDelete(err);
    ReleaseStageMetrics();
    stage_metrics_ = 0;
  }
}

void
ModelInstanceState::ReleaseStageMetrics()
{
  for (auto*& metric : stage_duration_metrics_) {
    if (metric != nullptr) {
      LOG_IF_ERROR(TRITONSERVER_MetricDelete(metric), "failed to delete metric");
      metric = nullptr;
    }
  }
  if (batch_requests_metric_ != nullptr) {
    LOG_IF_ERROR(
        TRITONSERVER_MetricDelete(batch_requests_metric_),
        "failed to delete metric");
    batch_requests_metric_ = nullptr;
  }
}

void
ModelInstanceState::ObserveStages(
    uint64_t exec_start_ns, uint64_t gather_end_ns, uint64_t compute_start_ns,
    uint64_t compute_end_ns, uint64_t exec_end_ns, uint32_t request_count)
{
  if (stage_metrics_ == 0) {
    return;
  }
  auto observe = [](TRITONSERVER_Metric* metric, uint64_t from, uint64_t to) {
    const double us = (to > from) ? (to - from) / 1000.0 : 0.0;
    LOG_IF_ERROR(
        TRITONSERVER_MetricObserve(metric, us), "failed to observe metric");
  };
  observe(stage_duration_metrics_[kGather], exec_start_ns, gather_end_ns);
  if (pipeline_) {
    observe(
        stage_duration_metrics_[kRunnerWait], gather_end_ns, compute_start_ns);
  }
  observe(stage_duration_metrics_[kRun], compute_start_ns, compute_end_ns);
  observe(stage_duration_metrics_[kOutput], compute_end_ns, exec_end_ns);
  LOG_IF_ERROR(
      TRITONSERVER_MetricObserve(batch_requests_metric_, request_count),
      "failed to observe metric");
}

size_t
ModelInstanceState::AcquireSlot()
{
  std::unique_lock<std::mutex> lk(slot_mu_);
  slot_cv_.wait(lk, [this] { return !free_slots_.empty(); });
  const size_t slot = free_slots_.back();
  free_slots_.pop_back();
  return slot;
}

void
ModelInstanceState::WaitForFreeSlot()
{
  std::unique_lock<std::mutex> lk(slot_mu_);
  slot_cv_.wait(lk, [this] { return !free_slots_.empty(); });
}

void
ModelInstanceState::ReleaseSlot(size_t slot)
{
  {
    std::lock_guard<std::mutex> lk(slot_mu_);
    free_slots_.push_back(slot);
  }
  slot_cv_.notify_one();
}

void
ModelInstanceState::RunPipelined(RunJob& job)
{
  // The current device is per thread, and this thread isn't Triton's.
#ifdef TRITON_ENABLE_GPU
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    static_cast<void>(cudaSetDevice(DeviceId()));
  }
#endif  // TRITON_ENABLE_GPU
#ifdef TRITON_ENABLE_ROCM
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    static_cast<void>(hipSetDevice(DeviceId()));
  }
#endif  // TRITON_ENABLE_ROCM

  const uint32_t request_count = static_cast<uint32_t>(job.requests.size());

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  if (!job.all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        job.responses, request_count, job.all_response_failed,
        OrtRun(&job.responses, request_count, job.binding));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  TRITONSERVER_Error* read_err = nullptr;
  if (!job.all_response_failed) {
    read_err = ReadOutputTensors(
        job.total_batch_size, job.requests.data(), request_count,
        &job.responses, job.binding, job.slot);
  }

  // Copies issued by the run or the output reads must finish before error
  // responses free their buffers, outputs return to the arena, or the
  // slot's pinned inputs are reused.
  SynchronizeStream();
  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      job.responses, request_count, job.all_response_failed, read_err);
  ReleaseOrtRunResources(job.binding);
  ReleaseSlot(job.slot);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  if (!job.all_response_failed) {
    ObserveStages(
        job.exec_start_ns, job.gather_end_ns, compute_start_ns, compute_end_ns,
        exec_end_ns, request_count);
  }

  CompleteExecution(
      job.requests.data(), request_count, std::move(job.responses),
      job.total_batch_size, job.all_response_failed, job.exec_start_ns,
      compute_start_ns, compute_end_ns, exec_end_ns);
}

void
ModelInstanceState::CompleteExecution(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>&& responses, size_t total_batch_size,
    bool all_response_failed, uint64_t exec_start_ns,
    uint64_t compute_start_ns, uint64_t compute_end_ns, uint64_t exec_end_ns)
{
  ResponseBatch batch;
  Completion& completion = batch.completion;
  completion.requests.assign(requests, requests + request_count);
  completion.success.reserve(request_count);
  for (auto* response : responses) {
    completion.success.push_back(response != nullptr);
  }
  completion.total_batch_size = total_batch_size;
  completion.batch_ok = !all_response_failed;
  completion.exec_start_ns = exec_start_ns;
  completion.compute_start_ns = compute_start_ns;
  completion.compute_end_ns = compute_end_ns;
  completion.exec_end_ns = exec_end_ns;
  batch.responses = std::move(responses);

  // ReadOutputTensors has already copied every output into the response
  // buffers and synchronized, so the responses no longer depend on any
  // per-run state and can be sent from another thread.
  if (response_q_.Running()) {
    response_q_.Push(std::move(batch));
  } else {
    SendAndComplete(batch);
  }
}

void
ModelInstanceState::SendAndComplete(ResponseBatch& batch)
{
  // Send all the responses that haven't already been sent because of
  // an earlier error. The success flags were captured beforehand, so the
  // completion still knows which requests succeeded.
  for (auto& response : batch.responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send onnxruntime backend response");
    }
  }
  DispatchCompletion(std::move(batch.completion));
}

void
ModelInstanceState::DispatchCompletion(Completion&& completion)
{
  if (completion_q_.Running()) {
    completion_q_.Push(std::move(completion));
  } else {
    FinishCompletion(completion);
  }
}

void
ModelInstanceState::FinishCompletion(Completion& completion)
{
  for (size_t r = 0; r < completion.requests.size(); ++r) {
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), completion.requests[r],
            completion.success[r] != 0, completion.exec_start_ns,
            completion.compute_start_ns, completion.compute_end_ns,
            completion.exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(
            completion.requests[r], TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (completion.batch_ok) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), completion.total_batch_size,
            completion.exec_start_ns, completion.compute_start_ns,
            completion.compute_end_ns, completion.exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::ValidateBooleanSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(
        InputInfos(session_, default_allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '") +
           tensor_name + "', but model does not provide that input")
              .c_str());
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (model_state_->MaxBatchSize() > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name + "' in model has dims " +
           ShapeToString(debatched_dims) + " but dims [1] is expected")
              .c_str());
    }

    if (ModelConfigDataTypeToOnnxDataType(tensor_datatype) !=
        iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name +
           "', the model expects data-type " +
           OnnxDataTypeName(iit->second.type_) +
           " but the model configuration specifies data-type " +
           tensor_datatype)
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateTypedSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(
        InputInfos(session_, default_allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '") +
           tensor_name + "', but model does not provide that input")
              .c_str());
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (model_state_->MaxBatchSize() > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name + "' in model has dims " +
           ShapeToString(debatched_dims) + " but dims [1] is expected")
              .c_str());
    }

    if (ModelConfigDataTypeToOnnxDataType(tensor_datatype) !=
        iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name +
           "', the model expects data-type " +
           OnnxDataTypeName(iit->second.type_) +
           " but the model configuration specifies data-type " +
           tensor_datatype)
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  std::set<std::string> input_tensor_names;
  RETURN_IF_ERROR(InputNames(session_, input_tensor_names));
  RETURN_IF_ERROR(
      InputInfos(session_, default_allocator_, input_tensor_infos_));

  std::set<std::string> overridable_initializer_tensor_names;
  RETURN_IF_ERROR(OverridableInitializerNames(
      session_, overridable_initializer_tensor_names));

  OnnxTensorInfoMap overridable_initializer_tensor_infos;
  RETURN_IF_ERROR(OverridableInitializerInfos(
      session_, default_allocator_, overridable_initializer_tensor_infos));

  if (input_tensor_infos_.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', configuration expects " + std::to_string(expected_input_cnt) +
         " inputs, model provides " +
         std::to_string(input_tensor_infos_.size()))
            .c_str());
  }

  // Merge 'overridable_xxx' into 'input_xxx' as they can be request inputs,
  // and all request inputs are checked against 'input_xxx'
  for (const auto& name : overridable_initializer_tensor_names) {
    input_tensor_names.emplace(name);
  }

  for (const auto& info : overridable_initializer_tensor_infos) {
    input_tensor_infos_[info.first] = info.second;
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    bool io_optional;
    RETURN_IF_ERROR(io.MemberAsBool("optional", &io_optional));

    if (io_optional && model_state_->MaxBatchSize() != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', optional input '" + io_name +
           "' is not supported for models that support batching")
              .c_str());
    }

    const auto& tensor_names = input_tensor_names;
    const auto& tensor_infos = input_tensor_infos_;
    auto iit = tensor_infos.find(io_name);
    if (iit == tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, tensor_names));
    }

    auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(io_dtype);
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (onnx_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for input '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromOnnxDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    triton::common::TritonJson::Value allow_ragged_batch_json;
    bool allow_ragged_batch = false;
    if (io.Find("allow_ragged_batch", &allow_ragged_batch_json)) {
      RETURN_IF_ERROR(allow_ragged_batch_json.AsBool(&allow_ragged_batch));
    }
    if (io_optional && allow_ragged_batch) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects model provides input with shape [-1] "
           "for ragged input '" +
           io_name + "', which is not supported for optional input")
              .c_str());
    }
    if (allow_ragged_batch) {
      const std::vector<int64_t>& model_shape = iit->second.dims_;
      // Make sure the input has shape [-1]
      if ((model_shape.size() != 1) || (model_shape[0] != WILDCARD_DIM)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unable to load model '") + model_state_->Name() +
             "', configuration expects model provides input with shape [-1]  "
             "for ragged input '" +
             io_name + "', model provides " + ShapeToString(model_shape))
                .c_str());
      }
    } else {
      // Only compare the dimensions if the tensor is not scalar
      if (iit->second.dims_.size() != 0) {
        RETURN_IF_ERROR(CompareDimsSupported(
            model_state_->Name(), io_name, iit->second.dims_, dims,
            model_state_->MaxBatchSize(), false /* compare_exact */));
      } else {
        // if max_batch_size == 0 and is a scalar tensor all the
        // dimensions specified must be equal to 1
        for (auto& dim : dims) {
          if (dim != 1) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("unable to load model '") + model_state_->Name() +
                 "', scalar tensor '" + io_name +
                 "', should only provide 1 in the model configuration when the "
                 "model doesn't support batching. Model configuration "
                 "provided: " +
                 ShapeToString(dims) + ".")
                    .c_str());
          }
        }
      }
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  std::set<std::string> output_tensor_names;
  RETURN_IF_ERROR(OutputNames(session_, output_tensor_names));

  RETURN_IF_ERROR(
      OutputInfos(session_, default_allocator_, output_tensor_infos_));

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = output_tensor_infos_.find(io_name);
    if (iit == output_tensor_infos_.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, output_tensor_names));
    }

    auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(io_dtype);
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for output '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (onnx_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for output '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromOnnxDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    // The batch output shape doesn't necessarily match the model
    if (model_state_->FindBatchOutput(io_name) == nullptr) {
      // Only compare the dimensions if the tensor is not scalar
      if (iit->second.dims_.size() != 0) {
        RETURN_IF_ERROR(CompareDimsSupported(
            model_state_->Name(), io_name, iit->second.dims_, dims,
            model_state_->MaxBatchSize(), true /* compare_exact */));
      } else {
        for (auto& dim : dims) {
          if (dim != 1) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("unable to load model '") + model_state_->Name() +
                 "', scalar tensor '" + io_name +
                 "', should only provide 1 in the model configuration when the "
                 "model doesn't support batching. Model configuration "
                 "provided: " +
                 ShapeToString(dims) + ".")
                    .c_str());
          }
        }

        // store the dimension for reference.
        scalar_outputs_[io_name] = dims;
      }
    }
  }

  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.MemberAsArray("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string state_name;
        RETURN_IF_ERROR(state.MemberAsString("name", &state_name));
        std::string state_dtype;
        RETURN_IF_ERROR(state.MemberAsString("data_type", &state_dtype));
        std::vector<int64_t> dims;
        RETURN_IF_ERROR(ParseShape(state, "dims", &dims));

        auto iit = output_tensor_infos_.find(state_name);
        if (iit == output_tensor_infos_.end()) {
          RETURN_IF_ERROR(CheckAllowedModelOutput(state, output_tensor_names));
        }

        auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(state_dtype);
        if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("unsupported datatype ") + state_dtype +
               " for output state '" + state_name + "' for model '" +
               model_state_->Name() + "'")
                  .c_str());
        } else if (onnx_data_type != iit->second.type_) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("unable to load model '") + model_state_->Name() +
               "', configuration expects datatype " + state_dtype +
               " for output state '" + state_name + "', model provides TYPE_" +
               TRITONSERVER_DataTypeString(
                   ConvertFromOnnxDataType(iit->second.type_)))
                  .c_str());
        }
      }
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() +
         " with " + std::to_string(request_count) + " requests")
            .c_str());
  }

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to ONNX Runtime backend for '" + Name() +
                  "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Use scoped class to clean up ORT tensors and other resources that
  // need to persist until ORT run completes.
  OrtIoBinding* binding = transient_binding_;
  struct ScopedCleanup {
    ScopedCleanup(ModelInstanceState* ctx, OrtIoBinding** binding)
        : ctx_(ctx), binding_(binding)
    {
    }
    ~ScopedCleanup()
    {
      if (ctx_ != nullptr) {
        ctx_->ReleaseOrtRunResources(*binding_);
      }
    }
    ModelInstanceState* ctx_;
    OrtIoBinding** binding_;
  } io_tensor_wrapper(this, &binding);

  const size_t slot = pipeline_ ? AcquireSlot() : 0;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());
  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      SetInputTensors(
          total_batch_size, requests, request_count, &responses, &collector,
          &cuda_copy, slot, &binding));

  if (!all_response_failed && !output_plan_built_) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        BuildOutputPlan(requests[0]));
  }

  if (!all_response_failed && !use_persistent_output_binding_) {
    // Rebinding to a device resets the bound value, so ORT allocates fresh
    // outputs rather than treating last run's as pre-allocated buffers.
    for (const auto& op : output_plan_) {
      RESPOND_ALL_AND_SET_TRUE_IF_ORT_ERROR(
          responses, request_count, all_response_failed,
          ort_api->BindOutputToDevice(
              binding, op.name_cstr, op.allocator_info));
    }
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_ROCM
  if (cuda_copy) {
    static_cast<void>(hipStreamSynchronize(static_cast<hipStream_t>(CudaStream())));
  }
#endif

  if (pipeline_) {
    // The runner takes the batch even if gathering failed: it owns the
    // binding's outputs from here on, frees the slot and completes the
    // requests on every path.
    io_tensor_wrapper.ctx_ = nullptr;
    RunJob job;
    job.slot = slot;
    job.binding = binding;
    job.total_batch_size = total_batch_size;
    job.all_response_failed = all_response_failed;
    job.exec_start_ns = exec_start_ns;
    job.requests.assign(requests, requests + request_count);
    job.responses = std::move(responses);
    SET_TIMESTAMP(job.gather_end_ns);
    runner_q_.Push(std::move(job));
    // Triton forms the next batch when Execute returns, so return only once
    // the runner can take it; until then requests keep joining the queue.
    WaitForFreeSlot();
    return;
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        OrtRun(&responses, request_count, binding));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  if (!all_response_failed) {
    TRITONSERVER_Error* read_err = ReadOutputTensors(
        total_batch_size, requests, request_count, &responses, binding, slot);
    if (read_err != nullptr) {
      // A failed read can return with copies still in flight.
      SynchronizeStream();
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed, read_err);
    }
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  if (!all_response_failed) {
    ObserveStages(
        exec_start_ns, compute_start_ns, compute_start_ns, compute_end_ns,
        exec_end_ns, request_count);
  }

  CompleteExecution(
      requests, request_count, std::move(responses), total_batch_size,
      all_response_failed, exec_start_ns, compute_start_ns, compute_end_ns,
      exec_end_ns);
}

TRITONSERVER_Error*
ModelInstanceState::OrtRun(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count, OrtIoBinding* binding)
{
  RETURN_IF_ORT_ERROR(ort_api->RunWithBinding(session_, runOptions_, binding));
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitPersistentInputBinding()
{
  use_persistent_input_binding_ = false;

  // Needs a batch dimension to key bindings on, and every input must be
  // present in every request with a row size known at init.
  if (model_state_->MaxBatchSize() <= 0) {
    return nullptr;
  }
  if (!StateForModel()->BatchInputs().empty()) {
    return nullptr;
  }
  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    return nullptr;
  }

  // Drive off the config input list, not input_tensor_infos_: the latter also
  // holds overridable initializers, which are not sent per request.
  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));

  std::vector<std::string> names;
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    bool io_optional = false;
    RETURN_IF_ERROR(io.MemberAsBool("optional", &io_optional));
    if (io_optional || StateForModel()->IsInputRagged(io_name)) {
      return nullptr;
    }

    auto iti = input_tensor_infos_.find(io_name);
    if (iti == input_tensor_infos_.end()) {
      return nullptr;
    }
    const auto& info = iti->second;
    if (info.type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING ||
        info.dims_.empty()) {
      return nullptr;
    }
    for (size_t d = 1; d < info.dims_.size(); d++) {
      if (info.dims_[d] < 0) {
        return nullptr;
      }
    }
    names.push_back(io_name);
  }
  if (names.empty()) {
    return nullptr;
  }

  const size_t max_batch = static_cast<size_t>(model_state_->MaxBatchSize());
  const auto build_start = std::chrono::steady_clock::now();
  TRITONSERVER_Error* err = [&]() -> TRITONSERVER_Error* {
    persistent_inputs_.reserve(names.size());
    for (const auto& name : names) {
      const auto& info = input_tensor_infos_.find(name)->second;
      int64_t row_elems = 1;
      for (size_t i = 1; i < info.dims_.size(); i++) {
        row_elems *= info.dims_[i];
      }

      persistent_inputs_.emplace_back();
      PersistentInput& pi = persistent_inputs_.back();
      pi.name = name;
      pi.dims = info.dims_;
      pi.type = info.type_;
      pi.row_bytes = row_elems * TRITONSERVER_DataTypeByteSize(
                                     ConvertFromOnnxDataType(info.type_));
      RETURN_IF_ERROR(AllocateStagingSlots(&pi, max_batch));
    }
    if (persistent_output_binding_ != 0) {
      RETURN_IF_ERROR(InitPersistentOutputs(max_batch));
    }

    persistent_bindings_.assign(
        staging_slots_, std::vector<PersistentBinding>(max_batch + 1));
    for (size_t slot = 0; slot < staging_slots_; ++slot) {
      for (size_t b = 1; b <= max_batch; ++b) {
        RETURN_IF_ERROR(
            BuildPersistentBinding(slot, b, &persistent_bindings_[slot][b]));
      }
    }
    return nullptr;
  }();
  if (err != nullptr) {
    ReleasePersistentState();
    return err;
  }
  const auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - build_start)
                            .count();

  use_persistent_input_binding_ = true;
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("persistent input binding enabled for ") +
       std::to_string(persistent_inputs_.size()) + " inputs at max batch " +
       std::to_string(max_batch) + ", staging slots " +
       std::to_string(staging_slots_) + ", persistent outputs " +
       std::to_string(persistent_outputs_.size()) + ", " +
       std::to_string(staging_slots_ * max_batch) + " bindings built in " +
       std::to_string(build_ms) + " ms")
          .c_str());
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::AllocateStagingSlots(
    PersistentInput* tensor, size_t max_batch)
{
  tensor->slots.resize(staging_slots_);
  for (auto& slot : tensor->slots) {
    RETURN_IF_ERROR(BackendMemory::Create(
        model_state_->TritonMemoryManager(),
        {BackendMemory::AllocationType::CPU_PINNED_POOL,
         BackendMemory::AllocationType::CPU},
        0 /* memory_type_id */, tensor->row_bytes * max_batch, &slot.memory));
    slot.base = slot.memory->MemoryPtr();
    slot.memory_type = slot.memory->MemoryType();
    slot.memory_type_id = slot.memory->MemoryTypeId();
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitPersistentOutputs(size_t max_batch)
{
  // Ineligible models keep binding outputs per run.
  auto skip = [this](const std::string& why) -> TRITONSERVER_Error* {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("persistent_output_binding ignored for '") + Name() +
         "': " + why)
            .c_str());
    return nullptr;
  };

  std::vector<PersistentInput> outputs;
  for (const auto& output : StateForModel()->ModelOutputs()) {
    const std::string& name = output.first;
    if ((output.second.first == -1) || (output.second.second != -1)) {
      return skip("output '" + name + "' is a state output");
    }
    if (StateForModel()->FindBatchOutput(name) != nullptr) {
      return skip("output '" + name + "' is a batch output");
    }
    auto iit = output_tensor_infos_.find(name);
    if (iit == output_tensor_infos_.end()) {
      return skip("output '" + name + "' has no tensor info");
    }
    const auto& info = iit->second;
    if ((info.type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) ||
        info.dims_.empty() || (info.dims_[0] >= 0) ||
        std::any_of(info.dims_.begin() + 1, info.dims_.end(), [](int64_t d) {
          return d < 0;
        })) {
      return skip(
          "output '" + name + "' is not batched with fixed dims, got " +
          ShapeToString(info.dims_));
    }

    outputs.emplace_back();
    PersistentInput& po = outputs.back();
    po.name = name;
    po.dims = info.dims_;
    po.type = info.type_;
    int64_t row_elems = 1;
    for (size_t i = 1; i < info.dims_.size(); i++) {
      row_elems *= info.dims_[i];
    }
    po.row_bytes = row_elems * TRITONSERVER_DataTypeByteSize(
                                   ConvertFromOnnxDataType(info.type_));
  }

  persistent_outputs_ = std::move(outputs);
  for (auto& po : persistent_outputs_) {
    RETURN_IF_ERROR(AllocateStagingSlots(&po, max_batch));
  }
  use_persistent_output_binding_ = true;
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::BuildPersistentBinding(
    size_t slot, size_t batch_size, PersistentBinding* out)
{
  // Built aside and moved in only when complete, so a failure can't leave
  // a partially bound binding in the table.
  PersistentBinding pb;
  TRITONSERVER_Error* err = [&]() -> TRITONSERVER_Error* {
    RETURN_IF_ORT_ERROR(ort_api->CreateIoBinding(session_, &pb.binding));
    pb.input_tensors.reserve(persistent_inputs_.size());
    std::vector<int64_t> shape;
    for (const auto& pi : persistent_inputs_) {
      const StagingSlot& stg = pi.slots[slot];
      shape.assign(pi.dims.begin(), pi.dims.end());
      shape[0] = static_cast<int64_t>(batch_size);
      pb.input_tensors.emplace_back(nullptr);
      RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          cpu_allocator_info_, stg.base, pi.row_bytes * batch_size,
          shape.data(), shape.size(), pi.type, &pb.input_tensors.back()));
      RETURN_IF_ORT_ERROR(ort_api->BindInput(
          pb.binding, pi.name.c_str(), pb.input_tensors.back()));
    }
    pb.output_tensors.reserve(persistent_outputs_.size());
    for (const auto& po : persistent_outputs_) {
      const StagingSlot& stg = po.slots[slot];
      shape.assign(po.dims.begin(), po.dims.end());
      shape[0] = static_cast<int64_t>(batch_size);
      pb.output_tensors.emplace_back(nullptr);
      RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          cpu_allocator_info_, stg.base, po.row_bytes * batch_size,
          shape.data(), shape.size(), po.type, &pb.output_tensors.back()));
      RETURN_IF_ORT_ERROR(ort_api->BindOutput(
          pb.binding, po.name.c_str(), pb.output_tensors.back()));
    }
    return nullptr;
  }();
  if (err != nullptr) {
    ReleasePersistentBinding(&pb);
    return err;
  }
  *out = std::move(pb);
  return nullptr;
}

void
ModelInstanceState::ReleasePersistentBinding(PersistentBinding* pb)
{
  for (auto* tensors : {&pb->input_tensors, &pb->output_tensors}) {
    for (auto* tensor : *tensors) {
      if (tensor != nullptr) {
        ort_api->ReleaseValue(tensor);
      }
    }
    tensors->clear();
  }
  if (pb->binding != nullptr) {
    ort_api->ReleaseIoBinding(pb->binding);
    pb->binding = nullptr;
  }
}

void
ModelInstanceState::ReleasePersistentState()
{
  for (auto& slot_bindings : persistent_bindings_) {
    for (auto& pb : slot_bindings) {
      ReleasePersistentBinding(&pb);
    }
  }
  persistent_bindings_.clear();
  for (auto* tensors : {&persistent_inputs_, &persistent_outputs_}) {
    for (auto& pt : *tensors) {
      for (auto& slot : pt.slots) {
        delete slot.memory;
      }
    }
    tensors->clear();
  }
  use_persistent_output_binding_ = false;
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensorsPersistent(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count, BackendInputCollector* collector,
    bool* cuda_copy, size_t slot, OrtIoBinding** binding)
{
  const auto& slot_bindings = persistent_bindings_[slot];
  RETURN_ERROR_IF_TRUE(
      total_batch_size == 0 || total_batch_size >= slot_bindings.size(),
      TRITONSERVER_ERROR_INTERNAL,
      std::string("no persistent binding for batch size ") +
          std::to_string(total_batch_size));
  *binding = slot_bindings[total_batch_size].binding;

  bool gathered = false;
  if (direct_input_gather_ != 0) {
    gathered = GatherPersistentInputsDirect(
        requests, request_count, total_batch_size, slot);
    if (!gathered && !direct_gather_warned_) {
      direct_gather_warned_ = true;
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("direct input gather fell back to the input collector "
                       "for '") +
           Name() + "'; later fallbacks are not logged")
              .c_str());
    }
  }
  if (gathered) {
    return nullptr;
  }

  for (const auto& pi : persistent_inputs_) {
    const StagingSlot& stg = pi.slots[slot];
    const size_t batchn_byte_size = pi.row_bytes * total_batch_size;

    // Gather into our buffer rather than one the collector allocates: the
    // stable data pointer is what lets the OrtValue and binding outlive the run.
    const char* dst = nullptr;
    size_t dst_bytes = 0;
    TRITONSERVER_MemoryType dst_type;
    int64_t dst_type_id;
    RETURN_IF_ERROR(collector->ProcessTensor(
        pi.name.c_str(), stg.base, batchn_byte_size,
        {{stg.memory_type, stg.memory_type_id}}, &dst, &dst_bytes, &dst_type,
        &dst_type_id));
    RETURN_ERROR_IF_TRUE(
        dst != stg.base, TRITONSERVER_ERROR_INTERNAL,
        std::string("input '") + pi.name +
            "' was not gathered into its persistent staging buffer");
  }

  *cuda_copy |= collector->Finalize();
  return nullptr;
}

bool
ModelInstanceState::GatherPersistentInputsDirect(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    size_t total_batch_size, size_t slot)
{
  gather_requests_ = requests;
  gather_request_count_ = request_count;
  gather_total_batch_size_ = total_batch_size;
  gather_slot_ = slot;
  gather_fj_.Run();
  for (const uint8_t ok : gather_part_ok_) {
    if (!ok) {
      return false;
    }
  }
  return true;
}

bool
ModelInstanceState::GatherPersistentInputs(size_t part)
{
  // Input by input with a local offset, so each part touches only its own
  // inputs and the writes into each staging buffer stay sequential.
  for (const size_t i : gather_parts_[part]) {
    const PersistentInput& pi = persistent_inputs_[i];
    const size_t capacity = pi.row_bytes * gather_total_batch_size_;
    char* base = pi.slots[gather_slot_].base;
    size_t offset = 0;
    for (uint32_t r = 0; r < gather_request_count_; ++r) {
      if (!GatherInputRows(pi, gather_requests_[r], base, capacity, &offset)) {
        return false;
      }
    }
    if (offset != capacity) {
      return false;
    }
  }
  return true;
}

bool
ModelInstanceState::GatherInputRows(
    const PersistentInput& pi, TRITONBACKEND_Request* request, char* base,
    size_t capacity, size_t* offset)
{
  const char* host_policy = HostPolicyName().c_str();
  TRITONBACKEND_Input* input;
  const int64_t* shape;
  uint32_t dims_count;
  uint64_t byte_size;
  uint32_t buffer_count;
  TRITONSERVER_Error* err =
      TRITONBACKEND_RequestInput(request, pi.name.c_str(), &input);
  if (err == nullptr) {
    err = TRITONBACKEND_InputPropertiesForHostPolicy(
        input, host_policy, nullptr, nullptr, &shape, &dims_count, &byte_size,
        &buffer_count);
  }
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
    return false;
  }
  if ((dims_count == 0) || (shape[0] < 0) ||
      (byte_size != static_cast<uint64_t>(shape[0]) * pi.row_bytes) ||
      (*offset + byte_size > capacity)) {
    return false;
  }

  size_t copied = 0;
  for (uint32_t b = 0; b < buffer_count; ++b) {
    const void* src;
    uint64_t src_bytes;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    err = TRITONBACKEND_InputBufferForHostPolicy(
        input, host_policy, b, &src, &src_bytes, &memory_type,
        &memory_type_id);
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      return false;
    }
    if ((memory_type == TRITONSERVER_MEMORY_GPU) ||
        (copied + src_bytes > byte_size)) {
      return false;
    }
    std::memcpy(base + *offset + copied, src, src_bytes);
    copied += src_bytes;
  }
  if (copied != byte_size) {
    return false;
  }
  *offset += byte_size;
  return true;
}

TRITONSERVER_Error*
ModelInstanceState::BuildOutputPlan(TRITONBACKEND_Request* request)
{
  TRITONSERVER_MemoryType preferred_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t preferred_memory_type_id = 0;
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    preferred_memory_type = TRITONSERVER_MEMORY_GPU;
    preferred_memory_type_id = DeviceId();
  }

  const auto& model_outputs = StateForModel()->ModelOutputs();
  output_plan_.clear();
  output_plan_.reserve(model_outputs.size());

  // [TODO] should collect only the outputs needed by some request.
  for (const auto& output : model_outputs) {
    OutputPlan op;
    op.name = &output.first;
    op.name_cstr = output.first.c_str();
    op.tensor_index = output.second.first;
    op.state_index = output.second.second;
    op.batch_output = StateForModel()->FindBatchOutput(output.first);

    // Strings bind to CPU. Everything else takes the destination's preferred
    // location, which the existing code already assumes is the same for every
    // request, so resolving it once from any request is equivalent.
    auto iit = output_tensor_infos_.find(output.first);
    if (iit == output_tensor_infos_.end()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string(
               "Error while retrieving output data type. Using cpu "
               "as preferred location for output: " +
               output.first)
               .c_str()));
    } else if (iit->second.type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      op.memory_type = preferred_memory_type;
      op.memory_type_id = preferred_memory_type_id;
      auto err = TRITONBACKEND_RequestOutputBufferProperties(
          request, op.name_cstr, /*byte_size*/ nullptr, &op.memory_type,
          &op.memory_type_id);
      if (err != nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string(
                 "Output Properties Unavailable. Using cpu as "
                 "preferred location for output: " +
                 output.first + " Error: " + TRITONSERVER_ErrorMessage(err))
                 .c_str()));
        TRITONSERVER_ErrorDelete(err);
        op.memory_type = TRITONSERVER_MEMORY_CPU;
        op.memory_type_id = 0;
      }
    }

    if (cuda_allocator_info_ == nullptr) {
      op.memory_type = TRITONSERVER_MEMORY_CPU;
      op.memory_type_id = 0;
    }
    op.allocator_info = (op.memory_type == TRITONSERVER_MEMORY_GPU)
                            ? cuda_allocator_info_
                            : cpu_allocator_info_;

    auto sit = scalar_outputs_.find(output.first);
    op.scalar_dims = (sit == scalar_outputs_.end()) ? nullptr : &sit->second;

    if ((model_state_->MaxBatchSize() > 0) && (op.batch_output == nullptr) &&
        (iit != output_tensor_infos_.end()) &&
        (iit->second.type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) &&
        (iit->second.dims_.size() > 1) && (iit->second.dims_[0] < 0) &&
        std::all_of(
            iit->second.dims_.begin() + 1, iit->second.dims_.end(),
            [](int64_t d) { return d >= 0; })) {
      op.dtype = ConvertFromOnnxDataType(iit->second.type_);
      op.fixed_dims = iit->second.dims_;
    }

    output_plan_.push_back(op);
  }

  size_t fixed_count = 0;
  output_fixed_ok_.assign(output_plan_.size(), 0);
  for (size_t idx = 0; idx < output_plan_.size(); ++idx) {
    output_fixed_ok_[idx] = !output_plan_[idx].fixed_dims.empty();
    fixed_count += output_fixed_ok_[idx];
  }
  output_batch_confirmed_.assign(
      static_cast<size_t>(std::max(0, model_state_->MaxBatchSize())) + 1, 0);

  output_plan_built_ = true;
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("output plan built for ") +
       std::to_string(output_plan_.size()) + " outputs, " +
       std::to_string(fixed_count) + " with fixed shapes")
          .c_str());
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, bool* cuda_copy, size_t slot,
    OrtIoBinding** binding)
{
  if (use_persistent_input_binding_) {
    return SetInputTensorsPersistent(
        total_batch_size, requests, request_count, collector, cuda_copy, slot,
        binding);
  }
  *binding = transient_binding_;

  const int max_batch_size = model_state_->MaxBatchSize();

  // When ORT_MIGRAPHX_INPUTS_ON_HOST is set, keep inputs host-resident (pinned)
  // for GPU instances instead of pre-copying them to device in the collector.
  // This hands the execution provider CPU-resident inputs so the MIGraphX
  // plugin EP's coalesced-input path (ORT_MIGRAPHX_COALESCE_IO) can batch every
  // per-input H2D into a single transfer, rather than Triton issuing one H2D
  // per input here. Defaults off (inputs bound on device, prior behavior).
  static const bool inputs_on_host{[] {
    const char* value{std::getenv("ORT_MIGRAPHX_INPUTS_ON_HOST")};
    return value != nullptr &&
           (std::string(value) == "1" || std::string(value) == "true");
  }()};

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_tensors_.emplace_back(nullptr);

    std::vector<int64_t> batchn_shape;
    // For a ragged input tensor, the tensor shape should be
    // the flatten shape of the whole batch
    if (StateForModel()->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        int64_t element_cnt = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            GetElementCount(input_shape, input_dims_count, &element_cnt));

        batchn_shape[0] += element_cnt;
      }
    }
    // The shape for the entire input batch, [total_batch_size, ...]
    else {
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (max_batch_size != 0) {
        batchn_shape[0] = total_batch_size;
      }
    }

    if (input_datatype != TRITONSERVER_TYPE_BYTES) {
      // The input must be in contiguous CPU memory. Use appropriate
      // allocator info to bind inputs to the right device. .i.e bind inputs
      // to GPU if they are being provided on GPU.
      const char* input_buffer;
      size_t batchn_byte_size;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>
          allowed_input_types;
      if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU && !inputs_on_host) {
        allowed_input_types = {
            {TRITONSERVER_MEMORY_GPU, DeviceId()},
            {TRITONSERVER_MEMORY_CPU_PINNED, 0},
            {TRITONSERVER_MEMORY_CPU, 0}};
      } else {
        // Host-resident inputs (pinned preferred). For GPU instances this is
        // gated by ORT_MIGRAPHX_INPUTS_ON_HOST so the EP performs a single
        // coalesced H2D instead of one copy per input in the collector.
        allowed_input_types = {
            {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};
      }

      RETURN_IF_ERROR(collector->ProcessTensor(
          input_name, nullptr, 0, allowed_input_types, &input_buffer,
          &batchn_byte_size, &memory_type, &memory_type_id));

      auto iti = input_tensor_infos_.find(input_name);
      if (iti == input_tensor_infos_.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                std::string(
                    "Failed to retrieve the ONNX input tensor info from '") +
                input_name + "'.")
                .c_str());
      }

      // Create ORT Tensor
      if (iti->second.dims_.size() == 0) {
        // scalar tensor
        RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
            memory_type == TRITONSERVER_MEMORY_GPU ? cuda_allocator_info_
                                                   : cpu_allocator_info_,
            (void*)input_buffer, batchn_byte_size, nullptr /* scalar */,
            0 /* number of dims */, ConvertToOnnxDataType(input_datatype),
            &input_tensors_.back()));
      } else {
        RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
            memory_type == TRITONSERVER_MEMORY_GPU ? cuda_allocator_info_
                                                   : cpu_allocator_info_,
            (void*)input_buffer, batchn_byte_size, batchn_shape.data(),
            batchn_shape.size(), ConvertToOnnxDataType(input_datatype),
            &input_tensors_.back()));
      }
      RETURN_IF_ORT_ERROR(
          ort_api->BindInput(*binding, input_name, input_tensors_.back()));
    } else {
      // For BYTES input, we need to convert the serialized string
      // representation into what is required for ORT. ORT expects a
      // vector of char*, one for each element. For each tensor we get
      // a copy of the data in a contiguous CPU buffer and then
      // in-place modify that from the Triton
      // <int32_len><bytes><int32_len><bytes>... serialization into a
      // <bytes><null-terminator><bytes><null-terminator>... serialization
      // and then initialize 'string_ptrs' to point to each <bytes>.
      std::vector<const char*> string_ptrs;

      SetStringInputTensor(
          requests, request_count, responses, input_name, &string_ptrs,
          cuda_copy);

      RETURN_IF_ORT_ERROR(ort_api->CreateTensorAsOrtValue(
          default_allocator_, batchn_shape.data(), batchn_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &input_tensors_.back()));
      RETURN_IF_ORT_ERROR(ort_api->FillStringTensor(
          input_tensors_.back(), string_ptrs.data(), string_ptrs.size()));
      RETURN_IF_ORT_ERROR(
          ort_api->BindInput(*binding, input_name, input_tensors_.back()));
    }
  }

  // Process batch input if any
  for (const auto& batch_input : StateForModel()->BatchInputs()) {
    std::vector<int64_t> shape;
    collector->BatchInputShape(batch_input, &shape);

    for (const auto& input_name : batch_input.TargetNames()) {
      input_tensors_.emplace_back(nullptr);

      const char* dst_buffer;
      size_t dst_buffer_byte_size;
      TRITONSERVER_MemoryType dst_memory_type;
      int64_t dst_memory_type_id;

      // Batch inputs are always created on CPU
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
          (*responses), responses->size(),
          collector->ProcessBatchInput(
              batch_input, nullptr, 0, {{TRITONSERVER_MEMORY_CPU, 0}},
              &dst_buffer, &dst_buffer_byte_size, &dst_memory_type,
              &dst_memory_type_id));

      // Create ORT Tensor
      RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          cpu_allocator_info_, (void*)dst_buffer, dst_buffer_byte_size,
          shape.data(), shape.size(),
          ConvertToOnnxDataType(batch_input.DataType()),
          &input_tensors_.back()));

      RETURN_IF_ORT_ERROR(ort_api->BindInput(
          *binding, input_name.c_str(), input_tensors_.back()));
    }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::SetStringInputTensor(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
    std::vector<const char*>* string_ptrs, bool* cuda_copy)
{
  size_t total_byte_size = 0;
  std::vector<size_t> expected_byte_sizes;
  std::vector<size_t> expected_element_cnts;
  expected_byte_sizes.reserve(request_count);
  expected_element_cnts.reserve(request_count);
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &((*responses)[ridx]),
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in));

    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        in, nullptr, nullptr, &input_shape, &input_dims_count, &input_byte_size,
        nullptr));

    // Skip input in this request if error response has already been sent.
    if ((*responses)[ridx] == nullptr) {
      expected_byte_sizes.push_back(0);
      expected_element_cnts.push_back(0);
    } else {
      int64_t element_cnt = 0;
      RETURN_IF_ERROR(
          GetElementCount(input_shape, input_dims_count, &element_cnt));
      expected_element_cnts.push_back(element_cnt);
      expected_byte_sizes.push_back(input_byte_size);
    }

    total_byte_size += expected_byte_sizes.back();
  }

  // For string input, the copy to contiguous buffer is needed because ORT
  // expects elements to be C strings thus we need to modify input buffer.
  // Reserve one more byte at the end of input_buffer to ensure last
  // element of String data can become valid C string.
  BackendMemory* input_memory;
  RETURN_IF_ERROR(BackendMemory::Create(
      model_state_->TritonMemoryManager(),
      {BackendMemory::AllocationType::CPU_PINNED_POOL,
       BackendMemory::AllocationType::CPU},
      0 /* memory_type_id */, total_byte_size + 1, &input_memory));
  input_tensor_memories_.push_back(input_memory);

  const TRITONSERVER_MemoryType mem_type = input_memory->MemoryType();
  char* input_buffer = input_memory->MemoryPtr();

  size_t buffer_offset = 0;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    TRITONSERVER_Error* err =
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in);
    if ((err == nullptr) && ((*responses)[ridx] != nullptr)) {
      uint32_t input_buffer_count;
      RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
          in, HostPolicyName().c_str(), nullptr, nullptr, nullptr, nullptr,
          nullptr, &input_buffer_count));

      size_t input_offset = 0;
      for (size_t idx = 0; idx < input_buffer_count; ++idx) {
        const void* src_buffer;
        size_t src_byte_size;
        TRITONSERVER_MemoryType src_memory_type;
        int64_t src_memory_type_id;
        err = TRITONBACKEND_InputBufferForHostPolicy(
            in, HostPolicyName().c_str(), idx, &src_buffer, &src_byte_size,
            &src_memory_type, &src_memory_type_id);
        if (err == nullptr) {
          if ((input_offset + src_byte_size) > expected_byte_sizes[ridx]) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("buffer size for input '") + input_name +
                 "' exceeds batch byte size " +
                 std::to_string(expected_byte_sizes[ridx]))
                    .c_str());
          } else {
            bool cuda_used = false;
            err = CopyBuffer(
                input_name, src_memory_type, src_memory_type_id, mem_type, 0,
                src_byte_size, src_buffer,
                input_buffer + buffer_offset + input_offset, CudaStream(),
                &cuda_used);
            *cuda_copy |= cuda_used;
          }
        }

        if (err == nullptr) {
          input_offset += src_byte_size;
        } else {
          break;
        }
      }
    }

    if (err != nullptr) {
      if ((*responses)[ridx] != nullptr) {
        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[ridx]), err);
      }

      TRITONSERVER_ErrorDelete(err);
    }

    buffer_offset += expected_byte_sizes[ridx];
  }

#ifdef TRITON_ENABLE_GPU
  // Synchronize to ensure the buffer is ready to be modified
  if (*cuda_copy) {
    cudaStreamSynchronize(CudaStream());
    *cuda_copy = false;
  }
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_ENABLE_ROCM
  // Synchronize to ensure the buffer is ready to be modified
  if (*cuda_copy) {
    static_cast<void>(hipStreamSynchronize(static_cast<hipStream_t>(CudaStream())));

    *cuda_copy = false;
  }
#endif  // TRITON_ENABLE_ROCM

  // Modify input buffer and set string expected by ORT
  SetStringInputBuffer(
      input_name, expected_byte_sizes, expected_element_cnts, responses,
      input_buffer, string_ptrs);
  input_buffer[total_byte_size] = 0;
  return nullptr;
}

void
ModelInstanceState::SetStringInputBuffer(
    const std::string& input_name,
    const std::vector<size_t>& expected_byte_sizes,
    const std::vector<size_t>& expected_element_cnts,
    std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
    std::vector<const char*>* string_ptrs)
{
  std::vector<std::pair<const char*, const uint32_t>> str_list;
  // offset for each response
  size_t buffer_copy_offset = 0;
  for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++) {
    const size_t expected_byte_size = expected_byte_sizes[idx];
    const size_t expected_element_cnt = expected_element_cnts[idx];

    if ((*responses)[idx] != nullptr) {
      char* data_content = input_buffer + buffer_copy_offset;
      TRITONSERVER_Error* err = ValidateStringBuffer(
          data_content, expected_byte_size, expected_element_cnt,
          input_name.c_str(), &str_list);
      // Set string values.
      for (const auto& [addr, len] : str_list) {
        // Make first byte of size info 0, so that if there is string data
        // in front of it, the data becomes valid C string.
        *const_cast<char*>(addr - sizeof(uint32_t)) = 0;
        string_ptrs->push_back(addr);
      }

      size_t element_cnt = str_list.size();
      if (err != nullptr) {
        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]), err);
        FillStringData(string_ptrs, expected_element_cnt - element_cnt);
      }
      str_list.clear();
    }
    buffer_copy_offset += expected_byte_size;
  }
}

void
ModelInstanceState::FillStringData(
    std::vector<const char*>* string_ptrs, size_t cnt)
{
  static const char* empty = "";
  for (size_t c = 0; c < cnt; c++) {
    string_ptrs->push_back(empty);
  }
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensor(
    std::vector<int64_t>& batchn_shape, TRITONSERVER_DataType& dtype,
    OrtValue* output_tensor, void** output_buffer,
    std::vector<std::vector<char>>& string_buffers,
    std::vector<size_t>& offsets)
{
  // Get output type and shape
  OrtTypeInfo* typeinfo;
  RETURN_IF_ORT_ERROR(ort_api->GetTypeInfo(output_tensor, &typeinfo));
  std::unique_ptr<OrtTypeInfo, TypeInfoDeleter> typeinfo_wrapper(typeinfo);

  const OrtTensorTypeAndShapeInfo* type_and_shape;
  RETURN_IF_ORT_ERROR(
      ort_api->CastTypeInfoToTensorInfo(typeinfo, &type_and_shape));

  size_t num_dims;
  RETURN_IF_ORT_ERROR(ort_api->GetDimensionsCount(type_and_shape, &num_dims));
  batchn_shape.resize(num_dims);
  RETURN_IF_ORT_ERROR(ort_api->GetDimensions(
      type_and_shape, batchn_shape.data(), batchn_shape.size()));

  ONNXTensorElementDataType type;
  RETURN_IF_ORT_ERROR(ort_api->GetTensorElementType(type_and_shape, &type));
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    int64_t element_count = 0;
    size_t total_length = 0;
    RETURN_IF_ERROR(GetElementCount(batchn_shape, &element_count));
    RETURN_IF_ORT_ERROR(
        ort_api->GetStringTensorDataLength(output_tensor, &total_length));

    string_buffers.emplace_back(std::vector<char>(total_length));
    auto content = string_buffers.back().data();
    offsets.reserve(element_count + 1);
    RETURN_IF_ORT_ERROR(ort_api->GetStringTensorContent(
        output_tensor, content, total_length, offsets.data(), element_count));
    // Mark "passed end byte offset"
    offsets[element_count] = total_length;

  } else {
    // Fixed size data type...
    RETURN_IF_ORT_ERROR(
        ort_api->GetTensorMutableData(output_tensor, output_buffer));
  }

  dtype = ConvertFromOnnxDataType(type);

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, OrtIoBinding* binding,
    size_t slot)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;

  OrtValue* const* outputs = nullptr;
  if (use_persistent_output_binding_) {
    RETURN_ERROR_IF_TRUE(
        persistent_outputs_.size() != output_plan_.size(),
        TRITONSERVER_ERROR_INTERNAL,
        std::string("persistent outputs do not match the output plan"));
    outputs =
        persistent_bindings_[slot][total_batch_size].output_tensors.data();
  } else {
    size_t output_count = 0;
    RETURN_IF_ORT_ERROR(ort_api->GetBoundOutputValues(
        binding, default_allocator_, &output_buffer_, &output_count));
    // Recorded before any error return so every value gets released.
    output_tensors_.assign(output_buffer_, output_buffer_ + output_count);
    if (output_count != output_plan_.size()) {
      RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("Retrieved output count is not equal to expected count.")));
    }
    outputs = output_buffer_;
  }

  if (use_persistent_output_binding_) {
    RETURN_IF_ERROR(PrepareDirectOutputs(
        total_batch_size, requests, request_count, responses));
  }

  const bool batch_confirmed =
      (total_batch_size < output_batch_confirmed_.size()) &&
      output_batch_confirmed_[total_batch_size];
  for (size_t idx = 0; idx < output_plan_.size(); idx++) {
    const auto& op = output_plan_[idx];
    OrtValue* output_tensor = outputs[idx];
    const std::string& name = *op.name;
    std::pair<TRITONSERVER_MemoryType, int64_t> alloc_perference{
        op.memory_type, op.memory_type_id};
    if (use_persistent_output_binding_) {
      const StagingSlot& stg = persistent_outputs_[idx].slots[slot];
      alloc_perference = {stg.memory_type, stg.memory_type_id};
    }

    const BatchOutput* batch_output = op.batch_output;
    if (batch_output == nullptr) {
      if (output_tensor == nullptr) {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("output tensor '") + name + "' is not found")
                .c_str()));
      }

      auto& batchn_shape = output_shape_scratch_;
      auto& string_buffers = output_string_scratch_;
      auto& offsets = output_offsets_scratch_;
      batchn_shape.clear();
      string_buffers.clear();
      offsets.clear();
      TRITONSERVER_DataType dtype;
      void* output_buffer;

      if (batch_confirmed && output_fixed_ok_[idx]) {
        batchn_shape.assign(op.fixed_dims.begin(), op.fixed_dims.end());
        batchn_shape[0] = static_cast<int64_t>(total_batch_size);
        dtype = op.dtype;
        RETURN_IF_ORT_ERROR(
            ort_api->GetTensorMutableData(output_tensor, &output_buffer));
      } else {
        RETURN_IF_ERROR(ReadOutputTensor(
            batchn_shape, dtype, output_tensor, &output_buffer, string_buffers,
            offsets));
        if (output_fixed_ok_[idx] &&
            ((dtype != op.dtype) ||
             (batchn_shape.size() != op.fixed_dims.size()) ||
             (batchn_shape[0] != static_cast<int64_t>(total_batch_size)) ||
             !std::equal(
                 batchn_shape.begin() + 1, batchn_shape.end(),
                 op.fixed_dims.begin() + 1))) {
          output_fixed_ok_[idx] = 0;
          LOG_MESSAGE(
              TRITONSERVER_LOG_VERBOSE,
              (std::string("output '") + name +
               "' does not match its fixed shape; reading its shape per run")
                  .c_str());
        }
      }

      // If the number of dimensions is equal to zero, it means that it is a
      // scalar and it would use the dimensions specified in the model
      // configuration.
      if (batchn_shape.size() == 0) {
        if (op.scalar_dims == nullptr) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "Failed to find the scalar output dimension for " + name +
                  " in the model configuration.")
                  .c_str());
        }
        batchn_shape = *op.scalar_dims;
      }

      if (op.tensor_index != -1) {
        if (dtype == TRITONSERVER_TYPE_BYTES) {
          auto content = string_buffers.back().data();
          cuda_copy |= SetStringOutputBuffer(
              name, content, offsets.data(), &batchn_shape, requests,
              request_count, responses);
        } else if (use_persistent_output_binding_ && output_fixed_ok_[idx]) {
          cuda_copy |= WriteDirectOutput(
              idx, dtype, batchn_shape, reinterpret_cast<char*>(output_buffer),
              alloc_perference.first, alloc_perference.second, responses);
        } else {
          responder.ProcessTensor(
              name, dtype, batchn_shape, reinterpret_cast<char*>(output_buffer),
              alloc_perference.first, alloc_perference.second);
        }
      }

      if (op.state_index != -1) {
        std::vector<TRITONBACKEND_State*> states;
        if (dtype == TRITONSERVER_TYPE_BYTES) {
          auto content = string_buffers.back().data();
          cuda_copy |= SetStringStateBuffer(
              name, content, offsets.data(), &batchn_shape, requests,
              request_count, responses);
        } else {
          states = responder.ProcessStateTensor(
              name, dtype, batchn_shape, reinterpret_cast<char*>(output_buffer),
              alloc_perference.first, alloc_perference.second);
        }

        // Update the states
        for (auto& state : states) {
          RETURN_IF_ERROR(TRITONBACKEND_StateUpdate(state));
        }
      }


    } else {
      char* output_buffer = nullptr;
      RETURN_IF_ORT_ERROR(
          ort_api->GetTensorMutableData(output_tensor, (void**)&output_buffer));
      responder.ProcessBatchOutput(
          name, *batch_output, output_buffer, alloc_perference.first,
          alloc_perference.second);
    }
  }

  if (total_batch_size < output_batch_confirmed_.size()) {
    output_batch_confirmed_[total_batch_size] = 1;
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
#ifdef TRITON_ENABLE_ROCM
  if (cuda_copy) {
    static_cast<void>(hipStreamSynchronize(static_cast<hipStream_t>(stream_)));
  }
#endif  // TRITON_ENABLE_ROCM
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::PrepareDirectOutputs(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  const size_t output_count = persistent_outputs_.size();
  request_batch_scratch_.assign(request_count, 0);
  request_wants_scratch_.assign(request_count * output_count, 0);

  int64_t batch_offset = 0;
  for (uint32_t r = 0; r < request_count; r++) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Input* input;
    const int64_t* shape;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, 0, &input));
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr));
    request_batch_scratch_[r] = shape[0];
    batch_offset += shape[0];
    RETURN_ERROR_IF_TRUE(
        batch_offset > static_cast<int64_t>(total_batch_size),
        TRITONSERVER_ERROR_INTERNAL,
        std::string("requests' batch sizes exceed the output batch size ") +
            std::to_string(total_batch_size));

    auto& response = (*responses)[r];
    if (response == nullptr) {
      continue;
    }
    uint32_t requested = 0;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, TRITONBACKEND_RequestOutputCount(request, &requested));
    uint8_t* wants = request_wants_scratch_.data() + r * output_count;
    // Requested names are unique model outputs, so a full count means all.
    if (requested == output_count) {
      std::fill(wants, wants + output_count, 1);
      continue;
    }
    for (uint32_t k = 0; (k < requested) && (response != nullptr); k++) {
      const char* name;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputName(request, k, &name));
      if (response == nullptr) {
        break;
      }
      for (size_t o = 0; o < output_count; o++) {
        if (persistent_outputs_[o].name == name) {
          wants[o] = 1;
          break;
        }
      }
    }
  }
  return nullptr;
}

bool
ModelInstanceState::WriteDirectOutput(
    size_t idx, TRITONSERVER_DataType dtype, std::vector<int64_t>& batchn_shape,
    const char* buffer, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, std::vector<TRITONBACKEND_Response*>* responses)
{
  const PersistentInput& po = persistent_outputs_[idx];
  const size_t output_count = persistent_outputs_.size();
  bool cuda_copy = false;
  size_t offset = 0;
  for (size_t r = 0; r < responses->size(); r++) {
    auto& response = (*responses)[r];
    const size_t byte_size = po.row_bytes * request_batch_scratch_[r];
    if ((response != nullptr) &&
        request_wants_scratch_[r * output_count + idx]) {
      batchn_shape[0] = request_batch_scratch_[r];
      TRITONBACKEND_Output* output;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_ResponseOutput(
                         response, &output, po.name.c_str(), dtype,
                         batchn_shape.data(), batchn_shape.size()));
      void* dst = nullptr;
      TRITONSERVER_MemoryType dst_type = memory_type;
      int64_t dst_type_id = memory_type_id;
      if (response != nullptr) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            &response, TRITONBACKEND_OutputBuffer(
                           output, &dst, byte_size, &dst_type, &dst_type_id));
      }
      if (response != nullptr) {
        bool cuda_used = false;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &response, CopyBuffer(
                           po.name, memory_type, memory_type_id, dst_type,
                           dst_type_id, byte_size, buffer + offset, dst,
                           stream_, &cuda_used));
        cuda_copy |= cuda_used;
      }
    }
    offset += byte_size;
  }
  return cuda_copy;
}

bool
ModelInstanceState::SetStringStateBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  return SetStringBuffer(
      name, content, offsets, batchn_shape, requests, request_count, responses,
      true /* state */);
}

bool
ModelInstanceState::SetStringOutputBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  return SetStringBuffer(
      name, content, offsets, batchn_shape, requests, request_count, responses,
      false /* state */);
}
bool
ModelInstanceState::SetStringBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, bool state)
{
  size_t element_idx = 0;
  bool cuda_copy = false;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    const auto& request = requests[ridx];
    auto& response = (*responses)[ridx];

    // batchn_shape holds the shape of the entire tensor batch. When
    // batching is enabled override the first batch dimension with each
    // requests batch size (reusing for efficiency).
    if (model_state_->MaxBatchSize() > 0) {
      TRITONBACKEND_Input* input;
      TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input);
      const int64_t* shape;
      TRITONBACKEND_InputProperties(
          input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
      (*batchn_shape)[0] = shape[0];
    }

    int64_t expected_element_cnt = 0;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, GetElementCount(*batchn_shape, &expected_element_cnt));

    // If 'request' requested this output then copy it from
    // 'content'. If it did not request this output then just skip it
    // in the 'content'.
    bool need_output = false;
    if (!state) {
      if (response != nullptr) {
        uint32_t output_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &response,
            TRITONBACKEND_RequestOutputCount(request, &output_count));
        if (response != nullptr) {
          for (uint32_t output_idx = 0; output_idx < output_count;
               output_idx++) {
            const char* req_output_name;
            RESPOND_AND_SET_NULL_IF_ERROR(
                &response, TRITONBACKEND_RequestOutputName(
                               request, output_idx, &req_output_name));
            if ((response != nullptr) && (req_output_name == name)) {
              need_output = true;
              break;
            }
          }
        }
      }
    } else {
      // need_output must be always set to true for state tensors.
      need_output = true;
    }

    if (need_output) {
      TRITONSERVER_Error* err;
      TRITONBACKEND_Output* response_output;
      TRITONBACKEND_State* response_state;
      if (!state) {
        err = TRITONBACKEND_ResponseOutput(
            response, &response_output, name.c_str(), TRITONSERVER_TYPE_BYTES,
            batchn_shape->data(), batchn_shape->size());
      } else {
        err = TRITONBACKEND_StateNew(
            &response_state, request, name.c_str(), TRITONSERVER_TYPE_BYTES,
            batchn_shape->data(), batchn_shape->size());
      }
      if (err == nullptr) {
        // Calculate expected byte size in advance using string offsets
        const size_t data_byte_size =
            offsets[element_idx + expected_element_cnt] - offsets[element_idx];
        const size_t expected_byte_size =
            data_byte_size + sizeof(uint32_t) * expected_element_cnt;

        TRITONSERVER_MemoryType actual_memory_type =
            TRITONSERVER_MEMORY_CPU_PINNED;
        int64_t actual_memory_type_id = 0;
        void* buffer;
        if (!state) {
          err = TRITONBACKEND_OutputBuffer(
              response_output, &buffer, expected_byte_size, &actual_memory_type,
              &actual_memory_type_id);
        } else {
          err = TRITONBACKEND_StateBuffer(
              response_state, &buffer, expected_byte_size, &actual_memory_type,
              &actual_memory_type_id);
        }
        if (err == nullptr) {
          bool cuda_used = false;
          size_t copied_byte_size = 0;
          for (size_t e = 0; e < expected_element_cnt; ++e) {
            const uint32_t len =
                offsets[element_idx + e + 1] - offsets[element_idx + e];
            // Prepend size of the string
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, sizeof(uint32_t),
                static_cast<const void*>(&len),
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += sizeof(uint32_t);

            // Copy raw string content
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, len, content + offsets[element_idx + e],
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += len;
          }
        }
      }

      RESPOND_AND_SET_NULL_IF_ERROR(&response, err);
      if (state) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            &response, TRITONBACKEND_StateUpdate(response_state));
      }
    }

    element_idx += expected_element_cnt;
  }

  return cuda_copy;
}

/////////////

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  // The backend configuration may contain information needed by the
  // ort backend, such as command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  TRITONSERVER_Error* err = nullptr;
  if (byte_size != 0) {
    err = backend_config.Parse(buffer, byte_size);
  }

  RETURN_IF_ERROR(err);

  // Onetime initialization for the onnxruntime loader.
  RETURN_IF_ERROR(OnnxLoader::Init(backend_config));

  std::unique_ptr<BackendConfiguration> lconfig(new BackendConfiguration());
  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;
    if (cmdline.Find("default-max-batch-size", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      int lvalue;
      RETURN_IF_ERROR(ParseIntValue(value_str, &lvalue));
      lconfig->default_max_batch_size_ = lvalue;
    }
  }
  // Check if device memory tracker is explicitly enabled
  if (DeviceMemoryTracker::EnableFromBackendConfig(backend_config)) {
    lconfig->enable_memory_tracker_ = DeviceMemoryTracker::Init();
  }
  TRITONSERVER_Error* metric_err = TRITONSERVER_MetricFamilyNew(
      &lconfig->stage_duration_family_, TRITONSERVER_METRIC_KIND_HISTOGRAM,
      "ort_backend_stage_duration_us",
      "Time per execution spent in each backend stage, in microseconds");
  if (metric_err == nullptr) {
    metric_err = TRITONSERVER_MetricFamilyNew(
        &lconfig->batch_requests_family_, TRITONSERVER_METRIC_KIND_HISTOGRAM,
        "ort_backend_batch_requests", "Requests per execution");
  }
  if (metric_err != nullptr) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("backend stage metrics unavailable: ") +
         TRITONSERVER_ErrorMessage(metric_err))
            .c_str());
    TRITONSERVER_ErrorDelete(metric_err);
  }
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(lconfig.get())));

  lconfig.release();
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  if (BackendConfiguration::RetrieveFrom(backend).enable_memory_tracker_) {
    DeviceMemoryTracker::Fini();
  }
  LOG_IF_ERROR(OnnxLoader::Stop(), "failed to stop OnnxLoader");
  void* state = nullptr;
  LOG_IF_ERROR(
      TRITONBACKEND_BackendState(backend, &state),
      "failed to get backend state");
  if (state != nullptr) {
    auto* config = reinterpret_cast<BackendConfiguration*>(state);
    for (auto* family :
         {config->stage_duration_family_, config->batch_requests_family_}) {
      if (family != nullptr) {
        LOG_IF_ERROR(
            TRITONSERVER_MetricFamilyDelete(family),
            "failed to delete metric family");
      }
    }
    delete config;
  }
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());


  // Utilizing DeviceMemoryTracker behavior that function calls with
  // 'nullptr' for usage will be no-ops.
  std::unique_ptr<DeviceMemoryTracker::MemoryUsage> lusage;
  if (BackendConfiguration::RetrieveFrom(model).enable_memory_tracker_) {
    lusage.reset(new DeviceMemoryTracker::MemoryUsage());
    DeviceMemoryTracker::TrackThreadMemoryUsage(lusage.get());
  }


  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  if (lusage) {
    DeviceMemoryTracker::UntrackThreadMemoryUsage(lusage.get());
    TRITONSERVER_BufferAttributes** ba_array;
    uint32_t ba_len = 0;
    RETURN_IF_ERROR(lusage->SerializeToBufferAttributes(&ba_array, &ba_len));
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelReportMemoryUsage(model, ba_array, ba_len));
  }
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // NOTE: If the corresponding TRITONBACKEND_BackendAttribute is enabled by the
  // backend for parallel model instance loading, the
  // TRITONBACKEND_ModelInstanceInitialize may be called concurrently.
  // Therefore, this function should be thread-safe.
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Utilizing DeviceMemoryTracker behavior that function calls with
  // 'nullptr' for usage will be no-ops.
  std::unique_ptr<DeviceMemoryTracker::MemoryUsage> lusage;
  if (BackendConfiguration::RetrieveFrom(instance).enable_memory_tracker_) {
    lusage.reset(new DeviceMemoryTracker::MemoryUsage());
    DeviceMemoryTracker::TrackThreadMemoryUsage(lusage.get());
  }


  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  if (lusage) {
    DeviceMemoryTracker::UntrackThreadMemoryUsage(lusage.get());
    TRITONSERVER_BufferAttributes** ba_array;
    uint32_t ba_len = 0;
    RETURN_IF_ERROR(lusage->SerializeToBufferAttributes(&ba_array, &ba_len));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportMemoryUsage(
        instance, ba_array, ba_len));
  }

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model ") + model_state->Name() + ", instance " +
         instance_state->Name() + ", executing " +
         std::to_string(request_count) + " requests")
            .c_str());
  }

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_GetBackendAttribute(
    TRITONBACKEND_Backend* backend,
    TRITONBACKEND_BackendAttribute* backend_attributes)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_GetBackendAttribute: setting attributes");
  // This backend can safely handle parallel calls to
  // TRITONBACKEND_ModelInstanceInitialize (thread-safe).
  RETURN_IF_ERROR(TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading(
      backend_attributes, true));

  return nullptr;
}

}  // extern "C"
}}}  // namespace triton::backend::onnxruntime
