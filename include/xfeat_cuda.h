#ifndef XFEAT_CUDA_H_
#define XFEAT_CUDA_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace xfeat_cuda {

void launchPreprocess(const unsigned char* source,
                      std::size_t source_step,
                      int source_height,
                      int source_width,
                      int source_channels,
                      float* destination,
                      int destination_height,
                      int destination_width,
                      cudaStream_t stream);

struct PostprocessConfig {
    int feature_height = 0;
    int feature_width = 0;
    int image_height = 0;
    int image_width = 0;
    int top_k = 0;
    int nms_kernel_size = 5;
    float detection_threshold = 0.05F;
    float softmax_temperature = 1.0F;
};

class Postprocessor {
public:
    Postprocessor(const PostprocessConfig& config, cudaStream_t stream);
    ~Postprocessor();

    Postprocessor(const Postprocessor&) = delete;
    Postprocessor& operator=(const Postprocessor&) = delete;

    // keypoints are returned as interleaved (x, y) model-input coordinates.
    void process(const float* feature_map,
                 const float* keypoint_logits,
                 const float* reliability_map,
                 std::vector<float>& keypoints,
                 std::vector<float>& descriptors,
                 std::vector<float>& scores);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace xfeat_cuda

#endif  // XFEAT_CUDA_H_
