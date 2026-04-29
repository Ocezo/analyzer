#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

namespace utils
{
// Shared result of the feature detection + matching + homography pipeline.
// Computed once in computeFeatureAlignment() and consumed by both analyzers.
struct FeatureMatchData
{
    cv::Mat resized1;
    cv::Mat resized2;
    cv::Mat gray1;
    cv::Mat gray2;
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::DMatch> good_matches;
    std::vector<unsigned char> inlier_mask;
    cv::Mat homography;
    bool descriptors_available = false;
};

// Runs the full shared pipeline: resize → grayscale → SIFT/ORB →
// BFMatcher + ratio test → findHomography (RANSAC).
FeatureMatchData computeFeatureAlignment(const cv::Mat& image1, const cv::Mat& image2);

cv::Mat loadColorImage(const std::string& path);
cv::Mat resizeToMaxSide(const cv::Mat& image, int max_side);
double compareColorHistograms(const cv::Mat& image1, const cv::Mat& image2);
double computeTextureSimilarity(const cv::Mat& image1, const cv::Mat& image2);
double computeStructuralSimilarity(const cv::Mat& image1, const cv::Mat& image2);
double clamp01(double value);
std::string formatScore(double value);
}  // namespace utils
