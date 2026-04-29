#include "utils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace utils
{
cv::Mat loadColorImage(const std::string& path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        throw std::runtime_error("Unable to load image: " + path);
    }
    return image;
}

cv::Mat resizeToMaxSide(const cv::Mat& image, int max_side)
{
    if (image.empty())
    {
        return image;
    }

    const int longest_side = std::max(image.cols, image.rows);
    if (longest_side <= max_side)
    {
        return image.clone();
    }

    const double scale = static_cast<double>(max_side) / static_cast<double>(longest_side);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    return resized;
}

double compareColorHistograms(const cv::Mat& image1, const cv::Mat& image2)
{
    if (image1.empty() || image2.empty())
    {
        return 0.0;
    }

    cv::Mat hsv1;
    cv::Mat hsv2;
    cv::cvtColor(image1, hsv1, cv::COLOR_BGR2HSV);
    cv::cvtColor(image2, hsv2, cv::COLOR_BGR2HSV);

    // Hue (0-180) with 36 bins, Saturation (0-256) with 32 bins.
    // Value channel is omitted: lighting changes strongly affect it but not scene identity.
    const int hs_bins[] = {36, 32};
    const float h_range[] = {0.0f, 180.0f};
    const float s_range[] = {0.0f, 256.0f};
    const float* hs_ranges[] = {h_range, s_range};
    const int hs_channels[] = {0, 1};

    cv::Mat hist1;
    cv::Mat hist2;
    cv::calcHist(&hsv1, 1, hs_channels, cv::Mat(), hist1, 2, hs_bins, hs_ranges, true, false);
    cv::calcHist(&hsv2, 1, hs_channels, cv::Mat(), hist2, 2, hs_bins, hs_ranges, true, false);

    cv::normalize(hist1, hist1, 1.0, 0.0, cv::NORM_L1);
    cv::normalize(hist2, hist2, 1.0, 0.0, cv::NORM_L1);

    const double bhattacharyya = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
    return clamp01(1.0 - bhattacharyya);
}

double computeTextureSimilarity(const cv::Mat& image1, const cv::Mat& image2)
{
    if (image1.empty() || image2.empty())
    {
        return 0.0;
    }

    cv::Mat normalized1;
    cv::Mat normalized2;
    cv::resize(image1, normalized1, cv::Size(256, 256), 0.0, 0.0, cv::INTER_AREA);
    cv::resize(image2, normalized2, cv::Size(256, 256), 0.0, 0.0, cv::INTER_AREA);

    cv::Mat laplacian1;
    cv::Mat laplacian2;
    cv::Laplacian(normalized1, laplacian1, CV_32F, 3);
    cv::Laplacian(normalized2, laplacian2, CV_32F, 3);

    cv::Scalar mean1;
    cv::Scalar stddev1;
    cv::Scalar mean2;
    cv::Scalar stddev2;
    cv::meanStdDev(laplacian1, mean1, stddev1);
    cv::meanStdDev(laplacian2, mean2, stddev2);

    const double energy1 = stddev1[0];
    const double energy2 = stddev2[0];
    const double delta = std::abs(energy1 - energy2);
    const double denom = std::max({energy1, energy2, 1.0});

    return clamp01(1.0 - delta / denom);
}

double computeStructuralSimilarity(const cv::Mat& image1, const cv::Mat& image2)
{
    if (image1.empty() || image2.empty())
    {
        return 0.0;
    }

    cv::Mat normalized1;
    cv::Mat normalized2;
    cv::resize(image1, normalized1, cv::Size(256, 256), 0.0, 0.0, cv::INTER_AREA);
    cv::resize(image2, normalized2, cv::Size(256, 256), 0.0, 0.0, cv::INTER_AREA);
    cv::GaussianBlur(normalized1, normalized1, cv::Size(9, 9), 1.5);
    cv::GaussianBlur(normalized2, normalized2, cv::Size(9, 9), 1.5);

    cv::Mat float1;
    cv::Mat float2;
    normalized1.convertTo(float1, CV_32F, 1.0 / 255.0);
    normalized2.convertTo(float2, CV_32F, 1.0 / 255.0);

    cv::Scalar mean1;
    cv::Scalar stddev1;
    cv::Scalar mean2;
    cv::Scalar stddev2;
    cv::meanStdDev(float1, mean1, stddev1);
    cv::meanStdDev(float2, mean2, stddev2);

    cv::Mat centered1 = float1 - mean1[0];
    cv::Mat centered2 = float2 - mean2[0];
    const double covariance = centered1.dot(centered2) / static_cast<double>(centered1.total());
    const double variance1 = stddev1[0] * stddev1[0];
    const double variance2 = stddev2[0] * stddev2[0];

    constexpr double c1 = 0.01 * 0.01;
    constexpr double c2 = 0.03 * 0.03;
    const double numerator = (2.0 * mean1[0] * mean2[0] + c1) * (2.0 * covariance + c2);
    const double denominator = (mean1[0] * mean1[0] + mean2[0] * mean2[0] + c1) *
                               (variance1 + variance2 + c2);

    if (denominator <= 0.0)
    {
        return 0.0;
    }

    const double ssim = numerator / denominator;
    return clamp01((ssim + 1.0) * 0.5);
}

double clamp01(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

std::string formatScore(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << clamp01(value);
    return stream.str();
}
}  // namespace utils
