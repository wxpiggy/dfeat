#ifndef LIGHTGLUE_H_
#define LIGHTGLUE_H_

#include <opencv2/core/types.hpp>

#include <memory>
#include <string>
#include <vector>

struct MatchPoint {
    int idx1;
    int idx2;
    float score;
};

class Lightglue {
public:
    Lightglue(const std::string& config_path, const std::string& engine_path);
    ~Lightglue();

    Lightglue(const Lightglue&) = delete;
    Lightglue& operator=(const Lightglue&) = delete;

    // keypoints are interleaved (x, y), descriptors are N x 64 row-major,
    // and image sizes are the real source image sizes (width, height).
    void matching(const std::vector<float>& keypoints1,
                  const std::vector<float>& keypoints2,
                  const std::vector<float>& descriptors1,
                  const std::vector<float>& descriptors2,
                  const cv::Size& image_size1,
                  const cv::Size& image_size2,
                  std::vector<MatchPoint>& matches);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // LIGHTGLUE_H_
