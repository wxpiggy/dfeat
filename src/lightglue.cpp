#include "lightglue.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class LightglueLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[LighterGlue TensorRT] " << message << '\n';
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

class CudaBuffer {
public:
    explicit CudaBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes_ > 0) {
            checkCuda(cudaMalloc(&data_, bytes_), "allocate LighterGlue CUDA buffer");
        }
    }
    ~CudaBuffer() { cudaFree(data_); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

void requireBinding(const nvinfer1::ICudaEngine& engine,
                    int index,
                    const char* name,
                    bool input,
                    nvinfer1::DataType type) {
    if (index < 0) {
        throw std::runtime_error(std::string("LighterGlue engine is missing binding: ") + name);
    }
    if (engine.bindingIsInput(index) != input) {
        throw std::runtime_error(std::string("LighterGlue binding has the wrong direction: ") + name);
    }
    if (engine.getBindingDataType(index) != type) {
        throw std::runtime_error(std::string("LighterGlue binding has an unexpected data type: ") +
                                 name);
    }
}

int outputMatchCount(const nvinfer1::Dims& dimensions) {
    if (dimensions.nbDims == 2 && dimensions.d[1] == 2) {
        return dimensions.d[0];
    }
    if (dimensions.nbDims == 3 && dimensions.d[0] == 1 && dimensions.d[2] == 2) {
        return dimensions.d[1];
    }
    return -1;
}

}  // namespace

struct Lightglue::Impl {
    Impl(const std::string& config_path, const std::string& engine_path) {
        try {
        const YAML::Node config = YAML::LoadFile(config_path);
        max_matches = config["max_matches"].as<int>();
        threshold = config["match_threshold"].as<float>();
        if (max_matches <= 0) {
            throw std::runtime_error("max_matches must be positive");
        }

        checkCuda(cudaStreamCreate(&stream), "create LighterGlue CUDA stream");
        initLibNvInferPlugins(&logger, "");
        const std::vector<char> engine_data = readBinaryFile(engine_path);
        runtime = nvinfer1::createInferRuntime(logger);
        if (!runtime) {
            throw std::runtime_error("Unable to create LighterGlue TensorRT runtime");
        }
        engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!engine) {
            throw std::runtime_error("Unable to deserialize LighterGlue TensorRT engine: " +
                                     engine_path);
        }
        context = engine->createExecutionContext();
        if (!context) {
            throw std::runtime_error("Unable to create LighterGlue TensorRT execution context");
        }

        image_size1_index = engine->getBindingIndex("image0_size");
        image_size2_index = engine->getBindingIndex("image1_size");
        keypoints1_index = engine->getBindingIndex("mkpts0");
        keypoints2_index = engine->getBindingIndex("mkpts1");
        descriptors1_index = engine->getBindingIndex("feats0");
        descriptors2_index = engine->getBindingIndex("feats1");
        matches_index = engine->getBindingIndex("matches");
        scores_index = engine->getBindingIndex("scores");

        requireBinding(*engine, image_size1_index, "image0_size", true,
                       nvinfer1::DataType::kFLOAT);
        requireBinding(*engine, image_size2_index, "image1_size", true,
                       nvinfer1::DataType::kFLOAT);
        requireBinding(*engine, keypoints1_index, "mkpts0", true, nvinfer1::DataType::kFLOAT);
        requireBinding(*engine, keypoints2_index, "mkpts1", true, nvinfer1::DataType::kFLOAT);
        requireBinding(*engine, descriptors1_index, "feats0", true,
                       nvinfer1::DataType::kFLOAT);
        requireBinding(*engine, descriptors2_index, "feats1", true,
                       nvinfer1::DataType::kFLOAT);
        requireBinding(*engine, matches_index, "matches", false, nvinfer1::DataType::kINT32);
        requireBinding(*engine, scores_index, "scores", false, nvinfer1::DataType::kFLOAT);
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
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

    LightglueLogger logger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    int image_size1_index = -1;
    int image_size2_index = -1;
    int keypoints1_index = -1;
    int keypoints2_index = -1;
    int descriptors1_index = -1;
    int descriptors2_index = -1;
    int matches_index = -1;
    int scores_index = -1;
    int max_matches = 0;
    float threshold = 0.0F;
};

Lightglue::Lightglue(const std::string& config_path, const std::string& engine_path)
    : impl_(std::make_unique<Impl>(config_path, engine_path)) {}

Lightglue::~Lightglue() = default;

void Lightglue::matching(const std::vector<float>& keypoints1,
                         const std::vector<float>& keypoints2,
                         const std::vector<float>& descriptors1,
                         const std::vector<float>& descriptors2,
                         const cv::Size& image_size1,
                         const cv::Size& image_size2,
                         std::vector<MatchPoint>& matches) {
    matches.clear();
    if (keypoints1.size() % 2 != 0 || keypoints2.size() % 2 != 0) {
        throw std::invalid_argument("LighterGlue keypoint arrays must contain (x, y) pairs");
    }
    const std::size_t count1 = keypoints1.size() / 2;
    const std::size_t count2 = keypoints2.size() / 2;
    if (descriptors1.size() != count1 * 64 || descriptors2.size() != count2 * 64) {
        throw std::invalid_argument("LighterGlue descriptor counts do not match keypoint counts");
    }
    if (image_size1.width <= 0 || image_size1.height <= 0 || image_size2.width <= 0 ||
        image_size2.height <= 0) {
        throw std::invalid_argument("LighterGlue image sizes must be positive");
    }
    if (count1 == 0 || count2 == 0) {
        return;
    }
    if (count1 > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        count2 > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Too many LighterGlue keypoints");
    }

    const int point_count1 = static_cast<int>(count1);
    const int point_count2 = static_cast<int>(count2);
    if (!impl_->context->setBindingDimensions(impl_->keypoints1_index,
                                               nvinfer1::Dims3{1, point_count1, 2}) ||
        !impl_->context->setBindingDimensions(impl_->keypoints2_index,
                                               nvinfer1::Dims3{1, point_count2, 2}) ||
        !impl_->context->setBindingDimensions(impl_->descriptors1_index,
                                               nvinfer1::Dims3{1, point_count1, 64}) ||
        !impl_->context->setBindingDimensions(impl_->descriptors2_index,
                                               nvinfer1::Dims3{1, point_count2, 64}) ||
        !impl_->context->allInputDimensionsSpecified()) {
        throw std::runtime_error(
            "LighterGlue rejected the dynamic keypoint counts; rebuild the engine with a compatible profile");
    }

    CudaBuffer image_size1_device(2 * sizeof(float));
    CudaBuffer image_size2_device(2 * sizeof(float));
    CudaBuffer keypoints1_device(keypoints1.size() * sizeof(float));
    CudaBuffer keypoints2_device(keypoints2.size() * sizeof(float));
    CudaBuffer descriptors1_device(descriptors1.size() * sizeof(float));
    CudaBuffer descriptors2_device(descriptors2.size() * sizeof(float));
    // There can be at most one mutual match for every point in image 0.
    CudaBuffer matches_device(count1 * 2 * sizeof(int));
    CudaBuffer scores_device(count1 * sizeof(float));

    const float host_image_size1[2] = {static_cast<float>(image_size1.width),
                                       static_cast<float>(image_size1.height)};
    const float host_image_size2[2] = {static_cast<float>(image_size2.width),
                                       static_cast<float>(image_size2.height)};
    checkCuda(cudaMemcpyAsync(image_size1_device.data(), host_image_size1, sizeof(host_image_size1),
                              cudaMemcpyHostToDevice, impl_->stream),
              "upload LighterGlue image0_size");
    checkCuda(cudaMemcpyAsync(image_size2_device.data(), host_image_size2, sizeof(host_image_size2),
                              cudaMemcpyHostToDevice, impl_->stream),
              "upload LighterGlue image1_size");
    checkCuda(cudaMemcpyAsync(keypoints1_device.data(), keypoints1.data(), keypoints1_device.bytes(),
                              cudaMemcpyHostToDevice, impl_->stream),
              "upload LighterGlue keypoints 0");
    checkCuda(cudaMemcpyAsync(keypoints2_device.data(), keypoints2.data(), keypoints2_device.bytes(),
                              cudaMemcpyHostToDevice, impl_->stream),
              "upload LighterGlue keypoints 1");
    checkCuda(cudaMemcpyAsync(descriptors1_device.data(), descriptors1.data(),
                              descriptors1_device.bytes(), cudaMemcpyHostToDevice, impl_->stream),
              "upload LighterGlue descriptors 0");
    checkCuda(cudaMemcpyAsync(descriptors2_device.data(), descriptors2.data(),
                              descriptors2_device.bytes(), cudaMemcpyHostToDevice, impl_->stream),
              "upload LighterGlue descriptors 1");

    std::vector<int> host_matches(count1 * 2, -1);
    std::vector<float> host_scores(count1, -std::numeric_limits<float>::infinity());
    checkCuda(cudaMemcpyAsync(matches_device.data(), host_matches.data(), matches_device.bytes(),
                              cudaMemcpyHostToDevice, impl_->stream),
              "initialize LighterGlue matches output");
    checkCuda(cudaMemcpyAsync(scores_device.data(), host_scores.data(), scores_device.bytes(),
                              cudaMemcpyHostToDevice, impl_->stream),
              "initialize LighterGlue scores output");

    std::vector<void*> bindings(static_cast<std::size_t>(impl_->engine->getNbBindings()), nullptr);
    bindings[impl_->image_size1_index] = image_size1_device.data();
    bindings[impl_->image_size2_index] = image_size2_device.data();
    bindings[impl_->keypoints1_index] = keypoints1_device.data();
    bindings[impl_->keypoints2_index] = keypoints2_device.data();
    bindings[impl_->descriptors1_index] = descriptors1_device.data();
    bindings[impl_->descriptors2_index] = descriptors2_device.data();
    bindings[impl_->matches_index] = matches_device.data();
    bindings[impl_->scores_index] = scores_device.data();
    if (!impl_->context->enqueueV2(bindings.data(), impl_->stream, nullptr)) {
        throw std::runtime_error("LighterGlue TensorRT inference failed");
    }

    checkCuda(cudaMemcpyAsync(host_matches.data(), matches_device.data(), matches_device.bytes(),
                              cudaMemcpyDeviceToHost, impl_->stream),
              "download LighterGlue matches");
    checkCuda(cudaMemcpyAsync(host_scores.data(), scores_device.data(), scores_device.bytes(),
                              cudaMemcpyDeviceToHost, impl_->stream),
              "download LighterGlue scores");
    checkCuda(cudaStreamSynchronize(impl_->stream), "synchronize LighterGlue inference");

    int slots = outputMatchCount(impl_->context->getBindingDimensions(impl_->matches_index));
    if (slots < 0) {
        // Data-dependent TensorRT outputs may retain a wildcard shape. The untouched
        // -inf sentinels distinguish the unused tail of the preallocated output.
        slots = point_count1;
    }
    if (slots > point_count1) {
        throw std::runtime_error("LighterGlue produced more matches than the output capacity");
    }
    matches.reserve(std::min(slots, impl_->max_matches));
    for (int index = 0; index < slots; ++index) {
        const float score = host_scores[index];
        if (!std::isfinite(score) || score < impl_->threshold) {
            continue;
        }
        const int index1 = host_matches[index * 2];
        const int index2 = host_matches[index * 2 + 1];
        if (index1 < 0 || index1 >= point_count1 || index2 < 0 || index2 >= point_count2) {
            continue;
        }
        matches.push_back({index1, index2, score});
        if (static_cast<int>(matches.size()) == impl_->max_matches) {
            break;
        }
    }
}
