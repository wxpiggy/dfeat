#include "xfeat_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace xfeat_cuda {
namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ float readGray(const unsigned char* source,
                          std::size_t step,
                          int height,
                          int width,
                          int channels,
                          int y,
                          int x) {
    y = max(0, min(y, height - 1));
    x = max(0, min(x, width - 1));
    const unsigned char* pixel = source + static_cast<std::size_t>(y) * step + x * channels;
    if (channels == 1) {
        return static_cast<float>(pixel[0]);
    }
    // XFeat's reference preprocessing uses the mean of the BGR channels.
    return (static_cast<float>(pixel[0]) + static_cast<float>(pixel[1]) +
            static_cast<float>(pixel[2])) / 3.0F;
}

__global__ void preprocessKernel(const unsigned char* source,
                                 std::size_t source_step,
                                 int source_height,
                                 int source_width,
                                 int source_channels,
                                 float* destination,
                                 int destination_height,
                                 int destination_width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destination_width || y >= destination_height) {
        return;
    }

    // Half-pixel bilinear resize, equivalent to align_corners=false.
    const float source_x = (static_cast<float>(x) + 0.5F) * source_width /
                               static_cast<float>(destination_width) -
                           0.5F;
    const float source_y = (static_cast<float>(y) + 0.5F) * source_height /
                               static_cast<float>(destination_height) -
                           0.5F;
    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float dx = source_x - x0;
    const float dy = source_y - y0;

    const float v00 = readGray(source, source_step, source_height, source_width,
                               source_channels, y0, x0);
    const float v01 = readGray(source, source_step, source_height, source_width,
                               source_channels, y0, x1);
    const float v10 = readGray(source, source_step, source_height, source_width,
                               source_channels, y1, x0);
    const float v11 = readGray(source, source_step, source_height, source_width,
                               source_channels, y1, x1);
    const float top = v00 + (v01 - v00) * dx;
    const float bottom = v10 + (v11 - v10) * dx;
    destination[y * destination_width + x] = (top + (bottom - top) * dy) / 255.0F;
}

__global__ void keypointHeatmapKernel(const float* logits,
                                      float* heatmap,
                                      int feature_height,
                                      int feature_width,
                                      float temperature) {
    const int feature_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (feature_x >= feature_width || feature_y >= feature_height) {
        return;
    }

    const int plane = feature_height * feature_width;
    const int offset = feature_y * feature_width + feature_x;
    float maximum = -3.402823466e+38F;
    for (int channel = 0; channel < 65; ++channel) {
        maximum = fmaxf(maximum, logits[channel * plane + offset] * temperature);
    }
    float denominator = 0.0F;
    for (int channel = 0; channel < 65; ++channel) {
        denominator += expf(logits[channel * plane + offset] * temperature - maximum);
    }

    const int image_width = feature_width * 8;
    for (int channel = 0; channel < 64; ++channel) {
        const int sub_y = channel / 8;
        const int sub_x = channel % 8;
        const int image_y = feature_y * 8 + sub_y;
        const int image_x = feature_x * 8 + sub_x;
        heatmap[image_y * image_width + image_x] =
            expf(logits[channel * plane + offset] * temperature - maximum) / denominator;
    }
}

__device__ float reliabilityAt(const float* reliability,
                               int feature_height,
                               int feature_width,
                               int full_height,
                               int full_width,
                               int x,
                               int y) {
    const float normalized_x = full_width > 1
                                   ? 2.0F * static_cast<float>(x) / (full_width - 1) - 1.0F
                                   : 0.0F;
    const float normalized_y = full_height > 1
                                   ? 2.0F * static_cast<float>(y) / (full_height - 1) - 1.0F
                                   : 0.0F;
    const float feature_x = ((normalized_x + 1.0F) * feature_width - 1.0F) / 2.0F;
    const float feature_y = ((normalized_y + 1.0F) * feature_height - 1.0F) / 2.0F;
    const int x0 = static_cast<int>(floorf(feature_x));
    const int y0 = static_cast<int>(floorf(feature_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float dx = feature_x - x0;
    const float dy = feature_y - y0;

    float v00 = 0.0F;
    float v01 = 0.0F;
    float v10 = 0.0F;
    float v11 = 0.0F;
    if (x0 >= 0 && x0 < feature_width && y0 >= 0 && y0 < feature_height) {
        v00 = reliability[y0 * feature_width + x0];
    }
    if (x1 >= 0 && x1 < feature_width && y0 >= 0 && y0 < feature_height) {
        v01 = reliability[y0 * feature_width + x1];
    }
    if (x0 >= 0 && x0 < feature_width && y1 >= 0 && y1 < feature_height) {
        v10 = reliability[y1 * feature_width + x0];
    }
    if (x1 >= 0 && x1 < feature_width && y1 >= 0 && y1 < feature_height) {
        v11 = reliability[y1 * feature_width + x1];
    }
    const float top = v00 + (v01 - v00) * dx;
    const float bottom = v10 + (v11 - v10) * dx;
    return top + (bottom - top) * dy;
}

__global__ void nmsKernel(const float* heatmap,
                          const float* reliability,
                          int image_height,
                          int image_width,
                          int feature_height,
                          int feature_width,
                          float threshold,
                          int radius,
                          int* count,
                          int* keypoints,
                          float* scores) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_width || y >= image_height) {
        return;
    }
    const int index = y * image_width + x;
    const float value = heatmap[index];
    if (value <= threshold) {
        return;
    }

    for (int dy = -radius; dy <= radius; ++dy) {
        const int neighbor_y = y + dy;
        if (neighbor_y < 0 || neighbor_y >= image_height) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            const int neighbor_x = x + dx;
            if (neighbor_x < 0 || neighbor_x >= image_width) {
                continue;
            }
            const int neighbor_index = neighbor_y * image_width + neighbor_x;
            const float neighbor = heatmap[neighbor_index];
            if (neighbor > value || (neighbor == value && neighbor_index < index)) {
                return;
            }
        }
    }

    const int output_index = atomicAdd(count, 1);
    keypoints[output_index * 2] = x;
    keypoints[output_index * 2 + 1] = y;
    scores[output_index] = value * reliabilityAt(reliability, feature_height, feature_width,
                                                  image_height, image_width, x, y);
}

__global__ void normalizeFeatureMapKernel(const float* source,
                                          float* destination,
                                          int feature_height,
                                          int feature_width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = feature_height * feature_width;
    if (index >= plane) {
        return;
    }
    float squared_norm = 0.0F;
    for (int channel = 0; channel < 64; ++channel) {
        const float value = source[channel * plane + index];
        squared_norm += value * value;
    }
    const float inverse_norm = rsqrtf(squared_norm + 1.0e-8F);
    for (int channel = 0; channel < 64; ++channel) {
        destination[channel * plane + index] = source[channel * plane + index] * inverse_norm;
    }
}

__device__ float cubicWeight(float value) {
    value = fabsf(value);
    constexpr float coefficient = -0.75F;
    if (value <= 1.0F) {
        return (coefficient + 2.0F) * value * value * value -
               (coefficient + 3.0F) * value * value + 1.0F;
    }
    if (value < 2.0F) {
        return coefficient * value * value * value - 5.0F * coefficient * value * value +
               8.0F * coefficient * value - 4.0F * coefficient;
    }
    return 0.0F;
}

__global__ void descriptorKernel(const float* feature_map,
                                 const int* keypoints,
                                 int count,
                                 int feature_height,
                                 int feature_width,
                                 int image_height,
                                 int image_width,
                                 float* descriptors) {
    const int keypoint_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (keypoint_index >= count) {
        return;
    }
    const int x = keypoints[keypoint_index * 2];
    const int y = keypoints[keypoint_index * 2 + 1];
    const float normalized_x = image_width > 1
                                   ? 2.0F * static_cast<float>(x) / (image_width - 1) - 1.0F
                                   : 0.0F;
    const float normalized_y = image_height > 1
                                   ? 2.0F * static_cast<float>(y) / (image_height - 1) - 1.0F
                                   : 0.0F;
    const float feature_x = ((normalized_x + 1.0F) * feature_width - 1.0F) / 2.0F;
    const float feature_y = ((normalized_y + 1.0F) * feature_height - 1.0F) / 2.0F;
    const int base_x = static_cast<int>(floorf(feature_x));
    const int base_y = static_cast<int>(floorf(feature_y));
    const float fraction_x = feature_x - base_x;
    const float fraction_y = feature_y - base_y;
    const int plane = feature_height * feature_width;

    float squared_norm = 0.0F;
    for (int channel = 0; channel < 64; ++channel) {
        float interpolated = 0.0F;
        for (int dy = -1; dy <= 2; ++dy) {
            const int sample_y = base_y + dy;
            const float weight_y = cubicWeight(fraction_y - dy);
            for (int dx = -1; dx <= 2; ++dx) {
                const int sample_x = base_x + dx;
                if (sample_x < 0 || sample_x >= feature_width || sample_y < 0 ||
                    sample_y >= feature_height) {
                    continue;
                }
                const float weight_x = cubicWeight(fraction_x - dx);
                interpolated += feature_map[channel * plane + sample_y * feature_width + sample_x] *
                                weight_x * weight_y;
            }
        }
        descriptors[keypoint_index * 64 + channel] = interpolated;
        squared_norm += interpolated * interpolated;
    }
    const float inverse_norm = rsqrtf(squared_norm + 1.0e-8F);
    for (int channel = 0; channel < 64; ++channel) {
        descriptors[keypoint_index * 64 + channel] *= inverse_norm;
    }
}

struct Candidate {
    int x;
    int y;
    float score;
};

}  // namespace

void launchPreprocess(const unsigned char* source,
                      std::size_t source_step,
                      int source_height,
                      int source_width,
                      int source_channels,
                      float* destination,
                      int destination_height,
                      int destination_width,
                      cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((destination_width + block.x - 1) / block.x,
                    (destination_height + block.y - 1) / block.y);
    preprocessKernel<<<grid, block, 0, stream>>>(source, source_step, source_height, source_width,
                                                 source_channels, destination, destination_height,
                                                 destination_width);
    checkCuda(cudaGetLastError(), "XFeat CUDA preprocessing");
}

struct Postprocessor::Impl {
    Impl(const PostprocessConfig& value, cudaStream_t cuda_stream)
        : config(value), stream(cuda_stream), candidate_capacity(value.image_height * value.image_width) {
        try {
        if (config.feature_height <= 0 || config.feature_width <= 0 || config.image_height <= 0 ||
            config.image_width <= 0 || config.top_k <= 0 || config.nms_kernel_size <= 0 ||
            config.nms_kernel_size % 2 == 0) {
            throw std::invalid_argument("Invalid XFeat postprocessing configuration");
        }
        checkCuda(cudaMalloc(&heatmap, candidate_capacity * sizeof(float)), "allocate heatmap");
        checkCuda(cudaMalloc(&candidate_keypoints, candidate_capacity * 2 * sizeof(int)),
                  "allocate candidate keypoints");
        checkCuda(cudaMalloc(&candidate_scores, candidate_capacity * sizeof(float)),
                  "allocate candidate scores");
        checkCuda(cudaMalloc(&candidate_count, sizeof(int)), "allocate candidate count");
        checkCuda(cudaMalloc(&selected_keypoints, config.top_k * 2 * sizeof(int)),
                  "allocate selected keypoints");
        checkCuda(cudaMalloc(&normalized_features,
                             64ULL * config.feature_height * config.feature_width * sizeof(float)),
                  "allocate normalized feature map");
        checkCuda(cudaMalloc(&descriptor_output, 64ULL * config.top_k * sizeof(float)),
                  "allocate descriptor output");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
        cudaFree(descriptor_output);
        descriptor_output = nullptr;
        cudaFree(normalized_features);
        normalized_features = nullptr;
        cudaFree(selected_keypoints);
        selected_keypoints = nullptr;
        cudaFree(candidate_count);
        candidate_count = nullptr;
        cudaFree(candidate_scores);
        candidate_scores = nullptr;
        cudaFree(candidate_keypoints);
        candidate_keypoints = nullptr;
        cudaFree(heatmap);
        heatmap = nullptr;
    }

    PostprocessConfig config;
    cudaStream_t stream = nullptr;
    int candidate_capacity = 0;
    float* heatmap = nullptr;
    int* candidate_keypoints = nullptr;
    float* candidate_scores = nullptr;
    int* candidate_count = nullptr;
    int* selected_keypoints = nullptr;
    float* normalized_features = nullptr;
    float* descriptor_output = nullptr;
};

Postprocessor::Postprocessor(const PostprocessConfig& config, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(config, stream)) {}

Postprocessor::~Postprocessor() = default;

void Postprocessor::process(const float* feature_map,
                            const float* keypoint_logits,
                            const float* reliability_map,
                            std::vector<float>& keypoints,
                            std::vector<float>& descriptors,
                            std::vector<float>& scores) {
    keypoints.clear();
    descriptors.clear();
    scores.clear();

    const auto& config = impl_->config;
    const dim3 heatmap_block(16, 16);
    const dim3 heatmap_grid((config.feature_width + heatmap_block.x - 1) / heatmap_block.x,
                            (config.feature_height + heatmap_block.y - 1) / heatmap_block.y);
    keypointHeatmapKernel<<<heatmap_grid, heatmap_block, 0, impl_->stream>>>(
        keypoint_logits, impl_->heatmap, config.feature_height, config.feature_width,
        config.softmax_temperature);
    checkCuda(cudaGetLastError(), "XFeat heatmap generation");

    checkCuda(cudaMemsetAsync(impl_->candidate_count, 0, sizeof(int), impl_->stream),
              "reset XFeat candidate count");
    const dim3 nms_block(16, 16);
    const dim3 nms_grid((config.image_width + nms_block.x - 1) / nms_block.x,
                        (config.image_height + nms_block.y - 1) / nms_block.y);
    nmsKernel<<<nms_grid, nms_block, 0, impl_->stream>>>(
        impl_->heatmap, reliability_map, config.image_height, config.image_width,
        config.feature_height, config.feature_width, config.detection_threshold,
        config.nms_kernel_size / 2, impl_->candidate_count, impl_->candidate_keypoints,
        impl_->candidate_scores);
    checkCuda(cudaGetLastError(), "XFeat NMS");

    int candidate_count = 0;
    checkCuda(cudaMemcpyAsync(&candidate_count, impl_->candidate_count, sizeof(int),
                              cudaMemcpyDeviceToHost, impl_->stream),
              "copy XFeat candidate count");
    checkCuda(cudaStreamSynchronize(impl_->stream), "synchronize XFeat NMS");
    candidate_count = std::min(candidate_count, impl_->candidate_capacity);
    if (candidate_count <= 0) {
        return;
    }

    std::vector<int> host_keypoints(static_cast<std::size_t>(candidate_count) * 2);
    std::vector<float> host_scores(candidate_count);
    checkCuda(cudaMemcpyAsync(host_keypoints.data(), impl_->candidate_keypoints,
                              host_keypoints.size() * sizeof(int), cudaMemcpyDeviceToHost,
                              impl_->stream),
              "copy XFeat candidate keypoints");
    checkCuda(cudaMemcpyAsync(host_scores.data(), impl_->candidate_scores,
                              host_scores.size() * sizeof(float), cudaMemcpyDeviceToHost,
                              impl_->stream),
              "copy XFeat candidate scores");
    checkCuda(cudaStreamSynchronize(impl_->stream), "synchronize XFeat candidates");

    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);
    for (int index = 0; index < candidate_count; ++index) {
        if (host_scores[index] > 0.0F && std::isfinite(host_scores[index])) {
            candidates.push_back(
                {host_keypoints[index * 2], host_keypoints[index * 2 + 1], host_scores[index]});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& first, const Candidate& second) {
        if (first.score != second.score) {
            return first.score > second.score;
        }
        if (first.y != second.y) {
            return first.y < second.y;
        }
        return first.x < second.x;
    });
    if (candidates.size() > static_cast<std::size_t>(config.top_k)) {
        candidates.resize(config.top_k);
    }
    const int selected_count = static_cast<int>(candidates.size());
    if (selected_count == 0) {
        return;
    }

    std::vector<int> selected(static_cast<std::size_t>(selected_count) * 2);
    keypoints.resize(static_cast<std::size_t>(selected_count) * 2);
    scores.resize(selected_count);
    for (int index = 0; index < selected_count; ++index) {
        selected[index * 2] = candidates[index].x;
        selected[index * 2 + 1] = candidates[index].y;
        keypoints[index * 2] = static_cast<float>(candidates[index].x);
        keypoints[index * 2 + 1] = static_cast<float>(candidates[index].y);
        scores[index] = candidates[index].score;
    }
    checkCuda(cudaMemcpyAsync(impl_->selected_keypoints, selected.data(),
                              selected.size() * sizeof(int), cudaMemcpyHostToDevice, impl_->stream),
              "upload selected XFeat keypoints");

    const int feature_pixels = config.feature_height * config.feature_width;
    constexpr int threads = 256;
    normalizeFeatureMapKernel<<<(feature_pixels + threads - 1) / threads, threads, 0,
                                impl_->stream>>>(feature_map, impl_->normalized_features,
                                                 config.feature_height, config.feature_width);
    checkCuda(cudaGetLastError(), "normalize XFeat feature map");
    descriptorKernel<<<(selected_count + threads - 1) / threads, threads, 0, impl_->stream>>>(
        impl_->normalized_features, impl_->selected_keypoints, selected_count, config.feature_height,
        config.feature_width, config.image_height, config.image_width, impl_->descriptor_output);
    checkCuda(cudaGetLastError(), "interpolate XFeat descriptors");

    descriptors.resize(static_cast<std::size_t>(selected_count) * 64);
    checkCuda(cudaMemcpyAsync(descriptors.data(), impl_->descriptor_output,
                              descriptors.size() * sizeof(float), cudaMemcpyDeviceToHost,
                              impl_->stream),
              "copy XFeat descriptors");
    checkCuda(cudaStreamSynchronize(impl_->stream), "synchronize XFeat postprocessing");
}

}  // namespace xfeat_cuda
