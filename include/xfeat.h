#ifndef XFEAT_H_
#define XFEAT_H_

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

struct XFeatResult {
    // Coordinates are interleaved (x, y) in the original image coordinate system.
    std::vector<float> keypoints;
    std::vector<float> descriptors;  // N x 64, row-major.
    std::vector<float> scores;
    cv::Size image_size;

    std::size_t size() const noexcept { return scores.size(); }
    void clear();
};

class XFeat {
public:
    XFeat(const std::string& config_path, const std::string& engine_path);
    ~XFeat();

    XFeat(const XFeat&) = delete;
    XFeat& operator=(const XFeat&) = delete;

    // Runs CUDA preprocessing, TensorRT inference and CUDA postprocessing.
    // No LibTorch objects are used in this path.
    void detectAndCompute(const cv::Mat& image, XFeatResult& result);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // XFEAT_H_
