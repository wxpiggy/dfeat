#include "lightglue.h"
#include "xfeat.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<cv::Point2f> toPoints(const std::vector<float>& values) {
    std::vector<cv::Point2f> points;
    points.reserve(values.size() / 2);
    for (std::size_t index = 0; index + 1 < values.size(); index += 2) {
        points.emplace_back(values[index], values[index + 1]);
    }
    return points;
}

cv::Mat toColor(const cv::Mat& image) {
    if (image.channels() == 3) {
        return image.clone();
    }
    cv::Mat color;
    cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    return color;
}

cv::Mat drawMatches(const cv::Mat& image1,
                    const cv::Mat& image2,
                    const XFeatResult& features1,
                    const XFeatResult& features2,
                    const std::vector<MatchPoint>& matches) {
    cv::Mat color1 = toColor(image1);
    cv::Mat color2 = toColor(image2);
    cv::Mat canvas(std::max(color1.rows, color2.rows), color1.cols + color2.cols, CV_8UC3,
                   cv::Scalar::all(0));
    color1.copyTo(canvas(cv::Rect(0, 0, color1.cols, color1.rows)));
    color2.copyTo(canvas(cv::Rect(color1.cols, 0, color2.cols, color2.rows)));

    const std::vector<cv::Point2f> points1 = toPoints(features1.keypoints);
    const std::vector<cv::Point2f> points2 = toPoints(features2.keypoints);
    std::vector<cv::Point2f> matched1;
    std::vector<cv::Point2f> matched2;
    matched1.reserve(matches.size());
    matched2.reserve(matches.size());
    for (const MatchPoint& match : matches) {
        if (match.idx1 >= 0 && match.idx1 < static_cast<int>(points1.size()) && match.idx2 >= 0 &&
            match.idx2 < static_cast<int>(points2.size())) {
            matched1.push_back(points1[match.idx1]);
            matched2.push_back(points2[match.idx2]);
        }
    }

    if (matched1.size() >= 4) {
        cv::Mat inlier_mask;
        const cv::Mat homography = cv::findHomography(matched1, matched2, cv::RANSAC, 3.0, inlier_mask);
        if (!homography.empty()) {
            const std::vector<cv::Point2f> corners = {
                {0.0F, 0.0F},
                {static_cast<float>(image1.cols - 1), 0.0F},
                {static_cast<float>(image1.cols - 1), static_cast<float>(image1.rows - 1)},
                {0.0F, static_cast<float>(image1.rows - 1)},
            };
            std::vector<cv::Point2f> warped;
            cv::perspectiveTransform(corners, warped, homography);
            // for (std::size_t index = 0; index < warped.size(); ++index) {
            //     cv::Point2f begin = warped[index] + cv::Point2f(static_cast<float>(image1.cols), 0.0F);
            //     cv::Point2f end = warped[(index + 1) % warped.size()] +
            //                       cv::Point2f(static_cast<float>(image1.cols), 0.0F);
            //     cv::line(canvas, begin, end, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            // }
        }
    }

    for (const MatchPoint& match : matches) {
        if (match.idx1 < 0 || match.idx1 >= static_cast<int>(points1.size()) || match.idx2 < 0 ||
            match.idx2 >= static_cast<int>(points2.size())) {
            continue;
        }
        const cv::Point2f point1 = points1[match.idx1];
        const cv::Point2f point2 =
            points2[match.idx2] + cv::Point2f(static_cast<float>(image1.cols), 0.0F);
        const float score = std::clamp(match.score, 0.0F, 1.0F);
        const cv::Scalar color(0, 255.0 * score, 255.0 * (1.0F - score));
        cv::circle(canvas, point1, 3, color, -1, cv::LINE_AA);
        cv::circle(canvas, point2, 3, color, -1, cv::LINE_AA);
        cv::line(canvas, point1, point2, color, 1, cv::LINE_AA);
    }

    std::ostringstream label;
    label << "keypoints " << features1.size() << "/" << features2.size() << ", matches "
          << matches.size();
    cv::putText(canvas, label.str(), cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, label.str(), cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return canvas;
}

void printUsage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " <image0> <image1> <xfeat.engine> <lighterglue.engine> [config.yaml] [output.png]\n"
              << "All keypoints are matched in the original image coordinate systems.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 7) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string image1_path = argv[1];
    const std::string image2_path = argv[2];
    const std::string xfeat_engine_path = argv[3];
    const std::string glue_engine_path = argv[4];
    const std::string config_path = argc >= 6 ? argv[5] : "../config/xfeat_lightglue.yaml";
    const std::string output_path = argc >= 7 ? argv[6] : "matching_result.png";
    for(int i = 0; i < 5; i ++){
    try {
        const cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
        const cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);
        if (image1.empty() || image2.empty()) {
            throw std::runtime_error("Unable to read one or both input images");
        }

        XFeat xfeat(config_path, xfeat_engine_path);
        Lightglue lightglue(config_path, glue_engine_path);
        XFeatResult features1;
        XFeatResult features2;
        std::vector<MatchPoint> matches;

        const auto begin = std::chrono::steady_clock::now();
        xfeat.detectAndCompute(image1, features1);
        
        std::cout << "fea1: " << features1.size() << std::endl;
        xfeat.detectAndCompute(image2, features2);
        
        lightglue.matching(features1.keypoints, features2.keypoints, features1.descriptors,
                           features2.descriptors, features1.image_size, features2.image_size,
                           matches);
        const auto end = std::chrono::steady_clock::now();
        


        const double milliseconds =
            std::chrono::duration<double, std::milli>(end - begin).count();
        std::cout << "image0=" << image1.cols << 'x' << image1.rows << ", image1=" << image2.cols
                  << 'x' << image2.rows << '\n'
                  << "keypoints=" << features1.size() << '/' << features2.size()
                  << ", matches=" << matches.size() << ", elapsed=" << std::fixed
                  << std::setprecision(2) << milliseconds << " ms\n"
                  << "saved: " << output_path << '\n';
                const cv::Mat visualization = drawMatches(image1, image2, features1, features2, matches);
        if (!cv::imwrite(output_path, visualization)) {
            throw std::runtime_error("Unable to write visualization: " + output_path);
        }
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 2;
    }
}
    return 0;
}
