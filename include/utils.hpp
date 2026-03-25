#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace utils
{
cv::Mat loadColorImage(const std::string& path);
cv::Mat resizeToMaxSide(const cv::Mat& image, int max_side);
double compareColorHistograms(const cv::Mat& image1, const cv::Mat& image2);
double computeTextureSimilarity(const cv::Mat& image1, const cv::Mat& image2);
double computeStructuralSimilarity(const cv::Mat& image1, const cv::Mat& image2);
double clamp01(double value);
std::string formatScore(double value);
}  // namespace utils
