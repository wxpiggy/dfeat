#include "xfeat.h"

#include "xfeat_cuda.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class XFeatLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[XFeat TensorRT] " << message << '\n';
        }
    }
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Unable to open TensorRT engine: " + path);
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path);
    }
    stream.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<std::size_t>(size));
    if (!stream.read(data.data(), size)) {
        throw std::runtime_error("Unable to read TensorRT engine: " + path);
    }
    return data;
}

void validateFloatBinding(const nvinfer1::ICudaEngine& engine, int index, const char* name) {
    if (index < 0) {
        throw std::runtime_error(std::string("XFeat engine is missing binding: ") + name);
    }
    if (engine.getBindingDataType(index) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(std::string("XFeat binding is not FP32: ") + name);
    }
}

}  // namespace

void XFeatResult::clear() {
    keypoints.clear();
    descriptors.clear();
    scores.clear();
    image_size = {};
}

struct XFeat::Impl {
    Impl(const std::string& config_path, const std::string& engine_path) {
        try {
        const YAML::Node config = YAML::LoadFile(config_path);
        model_height = config["image_height"].as<int>();
        model_width = config["image_width"].as<int>();
        top_k = config["max_keypoints"].as<int>();
        const float threshold = config["feat_threshold"].as<float>();
        const int kernel_size = config["kernel_size"].as<int>();
        const float softmax_temperature = config["softmaxTemp"].as<float>();

        if (model_height <= 0 || model_width <= 0 || top_k <= 0) {
            throw std::runtime_error("Invalid XFeat dimensions or max_keypoints in config");
        }
        checkCuda(cudaStreamCreate(&stream), "create XFeat CUDA stream");
        initLibNvInferPlugins(&logger, "");

        const std::vector<char> engine_data = readBinaryFile(engine_path);
        runtime = nvinfer1::createInferRuntime(logger);
        if (!runtime) {
            throw std::runtime_error("Unable to create XFeat TensorRT runtime");
        }
        engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!engine) {
            throw std::runtime_error("Unable to deserialize XFeat TensorRT engine: " + engine_path);
        }
        context = engine->createExecutionContext();
        if (!context) {
            throw std::runtime_error("Unable to create XFeat TensorRT execution context");
        }

        input_index = engine->getBindingIndex("images");
        features_index = engine->getBindingIndex("feats");
        logits_index = engine->getBindingIndex("keypoints");
        reliability_index = engine->getBindingIndex("heatmaps");
        validateFloatBinding(*engine, input_index, "images");
        validateFloatBinding(*engine, features_index, "feats");
        validateFloatBinding(*engine, logits_index, "keypoints");
        validateFloatBinding(*engine, reliability_index, "heatmaps");

        nvinfer1::Dims input_dims = engine->getBindingDimensions(input_index);
        if (input_dims.nbDims != 4) {
            throw std::runtime_error("XFeat input must have NCHW dimensions");
        }
        if (input_dims.d[2] > 0) {
            model_height = input_dims.d[2];
        }
        if (input_dims.d[3] > 0) {
            model_width = input_dims.d[3];
        }
        bool has_dynamic_input = false;
        for (int index = 0; index < input_dims.nbDims; ++index) {
            has_dynamic_input = has_dynamic_input || input_dims.d[index] < 0;
        }
        if (has_dynamic_input) {
            if (!context->setBindingDimensions(
                    input_index, nvinfer1::Dims4{1, 1, model_height, model_width})) {
                throw std::runtime_error("Unable to set dynamic XFeat input dimensions");
            }
        }
        if (!context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Not all XFeat input dimensions are specified");
        }

        const nvinfer1::Dims feature_dims = context->getBindingDimensions(features_index);
        const nvinfer1::Dims logit_dims = context->getBindingDimensions(logits_index);
        const nvinfer1::Dims reliability_dims = context->getBindingDimensions(reliability_index);
        if (feature_dims.nbDims != 4 || feature_dims.d[1] != 64 || feature_dims.d[2] <= 0 ||
            feature_dims.d[3] <= 0 || logit_dims.nbDims != 4 || logit_dims.d[1] != 65 ||
            reliability_dims.nbDims != 4 || reliability_dims.d[1] != 1) {
            throw std::runtime_error("Unexpected XFeat output dimensions");
        }
        feature_height = feature_dims.d[2];
        feature_width = feature_dims.d[3];
        if (logit_dims.d[2] != feature_height || logit_dims.d[3] != feature_width ||
            reliability_dims.d[2] != feature_height || reliability_dims.d[3] != feature_width ||
            feature_height * 8 != model_height || feature_width * 8 != model_width) {
            throw std::runtime_error("Inconsistent XFeat engine input/output dimensions");
        }

        checkCuda(cudaMalloc(&input, static_cast<std::size_t>(model_height) * model_width * sizeof(float)),
                  "allocate XFeat input");
        checkCuda(cudaMalloc(&features,
                             64ULL * feature_height * feature_width * sizeof(float)),
                  "allocate XFeat features");
        checkCuda(cudaMalloc(&logits, 65ULL * feature_height * feature_width * sizeof(float)),
                  "allocate XFeat keypoint logits");
        checkCuda(cudaMalloc(&reliability,
                             static_cast<std::size_t>(feature_height) * feature_width * sizeof(float)),
                  "allocate XFeat reliability map");

        xfeat_cuda::PostprocessConfig postprocess_config;
        postprocess_config.feature_height = feature_height;
        postprocess_config.feature_width = feature_width;
        postprocess_config.image_height = model_height;
        postprocess_config.image_width = model_width;
        postprocess_config.top_k = top_k;
        postprocess_config.nms_kernel_size = kernel_size;
        postprocess_config.detection_threshold = threshold;
        postprocess_config.softmax_temperature = softmax_temperature;
        postprocessor = std::make_unique<xfeat_cuda::Postprocessor>(postprocess_config, stream);
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
        postprocessor.reset();
        cudaFree(device_image);
        device_image = nullptr;
        cudaFree(reliability);
        reliability = nullptr;
        cudaFree(logits);
        logits = nullptr;
        cudaFree(features);
        features = nullptr;
        cudaFree(input);
        input = nullptr;
        if (context) {
            context->destroy();
            context = nullptr;
        }
        if (engine) {
            engine->destroy();
            engine = nullptr;
        }
        if (runtime) {
            runtime->destroy();
            runtime = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    void ensureImageCapacity(std::size_t bytes) {
        if (bytes <= device_image_capacity) {
            return;
        }
        if (device_image) {
            checkCuda(cudaFree(device_image), "release old XFeat image buffer");
            device_image = nullptr;
            device_image_capacity = 0;
        }
        checkCuda(cudaMalloc(&device_image, bytes), "allocate XFeat image buffer");
        device_image_capacity = bytes;
    }

    XFeatLogger logger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    int input_index = -1;
    int features_index = -1;
    int logits_index = -1;
    int reliability_index = -1;
    int model_height = 0;
    int model_width = 0;
    int feature_height = 0;
    int feature_width = 0;
    int top_k = 0;
    unsigned char* device_image = nullptr;
    std::size_t device_image_capacity = 0;
    float* input = nullptr;
    float* features = nullptr;
    float* logits = nullptr;
    float* reliability = nullptr;
    std::unique_ptr<xfeat_cuda::Postprocessor> postprocessor;
};

XFeat::XFeat(const std::string& config_path, const std::string& engine_path)
    : impl_(std::make_unique<Impl>(config_path, engine_path)) {}

XFeat::~XFeat() = default;

void XFeat::detectAndCompute(const cv::Mat& image, XFeatResult& result) {
    result.clear();
    if (image.empty()) {
        throw std::invalid_argument("XFeat input image is empty");
    }
    if (image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3)) {
        throw std::invalid_argument("XFeat expects an 8-bit one- or three-channel image");
    }

    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * image.elemSize();
    const std::size_t image_bytes = row_bytes * image.rows;
    impl_->ensureImageCapacity(image_bytes);
    checkCuda(cudaMemcpy2DAsync(impl_->device_image, row_bytes, image.data, image.step, row_bytes,
                                image.rows, cudaMemcpyHostToDevice, impl_->stream),
              "upload XFeat image");
              const auto begin = std::chrono::steady_clock::now();
    xfeat_cuda::launchPreprocess(impl_->device_image, row_bytes, image.rows, image.cols,
                                 image.channels(), impl_->input, impl_->model_height,
                                 impl_->model_width, impl_->stream);

    std::vector<void*> bindings(static_cast<std::size_t>(impl_->engine->getNbBindings()), nullptr);
    bindings[impl_->input_index] = impl_->input;
    bindings[impl_->features_index] = impl_->features;
    bindings[impl_->logits_index] = impl_->logits;
    bindings[impl_->reliability_index] = impl_->reliability;
    
    if (!impl_->context->enqueueV2(bindings.data(), impl_->stream, nullptr)) {
        throw std::runtime_error("XFeat TensorRT inference failed");
    }

    impl_->postprocessor->process(impl_->features, impl_->logits, impl_->reliability,
                                  result.keypoints, result.descriptors, result.scores);
                const auto end = std::chrono::steady_clock::now();
        


        const double milliseconds =
            std::chrono::duration<double, std::milli>(end - begin).count();
            std::cout << milliseconds << "ms" << std::endl;
                                  const float scale_x = static_cast<float>(image.cols) / impl_->model_width;
    const float scale_y = static_cast<float>(image.rows) / impl_->model_height;
    for (std::size_t index = 0; index < result.size(); ++index) {
        result.keypoints[index * 2] *= scale_x;
        result.keypoints[index * 2 + 1] *= scale_y;
    }
    result.image_size = image.size();
}
