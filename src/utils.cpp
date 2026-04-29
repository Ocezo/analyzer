#include "utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace
{
constexpr int kMaxSide = 1400;
constexpr int kMaxFeatures = 2500;
constexpr float kRatioTest = 0.75f;
constexpr double kHomographyRansacThreshold = 5.0;
}  // namespace

namespace utils
{
FeatureMatchData computeFeatureAlignment(const cv::Mat& image1, const cv::Mat& image2)
{
    FeatureMatchData data;
    if (image1.empty() || image2.empty())
    {
        return data;
    }

    data.resized1 = resizeToMaxSide(image1, kMaxSide);
    data.resized2 = resizeToMaxSide(image2, kMaxSide);
    cv::cvtColor(data.resized1, data.gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(data.resized2, data.gray2, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::Feature2D> detector;
    int matcher_norm = cv::NORM_L2;
    try
    {
        detector = cv::SIFT::create(kMaxFeatures);
    }
    catch (const cv::Exception&)
    {
        detector = cv::ORB::create(kMaxFeatures);
        matcher_norm = cv::NORM_HAMMING;
    }

    cv::Mat descriptors1;
    cv::Mat descriptors2;
    detector->detectAndCompute(data.gray1, cv::noArray(), data.keypoints1, descriptors1);
    detector->detectAndCompute(data.gray2, cv::noArray(), data.keypoints2, descriptors2);

    if (descriptors1.empty() || descriptors2.empty())
    {
        return data;
    }

    data.descriptors_available = true;

    cv::BFMatcher matcher(matcher_norm);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    data.good_matches.reserve(knn_matches.size());
    for (const auto& pair : knn_matches)
    {
        if (pair.size() >= 2 && pair[0].distance < kRatioTest * pair[1].distance)
        {
            data.good_matches.push_back(pair[0]);
        }
    }

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(data.good_matches.size());
    points2.reserve(data.good_matches.size());
    for (const auto& match : data.good_matches)
    {
        points1.push_back(data.keypoints1[match.queryIdx].pt);
        points2.push_back(data.keypoints2[match.trainIdx].pt);
    }

    if (points1.size() >= 4)
    {
        data.homography = cv::findHomography(
            points1, points2, cv::RANSAC, kHomographyRansacThreshold, data.inlier_mask);
    }

    return data;
}


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

    // HS histogram (Hue × Saturation) on saturated pixels only: pixels with
    // low saturation (grey, white, black) have an unstable/undefined Hue and
    // would inflate the Bhattacharyya distance even for identical scenes.
    // Saturation threshold = 30/255 in 8-bit → pixels must be at least faintly coloured.
    cv::Mat sat_mask1;
    cv::Mat sat_mask2;
    cv::extractChannel(hsv1, sat_mask1, 1);
    cv::extractChannel(hsv2, sat_mask2, 1);
    cv::threshold(sat_mask1, sat_mask1, 30, 255, cv::THRESH_BINARY);
    cv::threshold(sat_mask2, sat_mask2, 30, 255, cv::THRESH_BINARY);

    const int hs_bins[] = {36, 32};
    const float h_range[] = {0.0f, 180.0f};
    const float s_range[] = {0.0f, 256.0f};
    const float* hs_ranges[] = {h_range, s_range};
    const int hs_channels[] = {0, 1};

    cv::Mat hs_hist1;
    cv::Mat hs_hist2;
    cv::calcHist(&hsv1, 1, hs_channels, sat_mask1, hs_hist1, 2, hs_bins, hs_ranges, true, false);
    cv::calcHist(&hsv2, 1, hs_channels, sat_mask2, hs_hist2, 2, hs_bins, hs_ranges, true, false);
    cv::normalize(hs_hist1, hs_hist1, 1.0, 0.0, cv::NORM_L1);
    cv::normalize(hs_hist2, hs_hist2, 1.0, 0.0, cv::NORM_L1);
    const double hs_bhattacharyya = cv::compareHist(hs_hist1, hs_hist2, cv::HISTCMP_BHATTACHARYYA);

    // V (brightness) histogram on all pixels: captures overall luminance profile.
    const int v_bins[] = {32};
    const float v_range[] = {0.0f, 256.0f};
    const float* v_ranges[] = {v_range};
    const int v_channels[] = {2};

    cv::Mat v_hist1;
    cv::Mat v_hist2;
    cv::calcHist(&hsv1, 1, v_channels, cv::Mat(), v_hist1, 1, v_bins, v_ranges, true, false);
    cv::calcHist(&hsv2, 1, v_channels, cv::Mat(), v_hist2, 1, v_bins, v_ranges, true, false);
    cv::normalize(v_hist1, v_hist1, 1.0, 0.0, cv::NORM_L1);
    cv::normalize(v_hist2, v_hist2, 1.0, 0.0, cv::NORM_L1);
    const double v_bhattacharyya = cv::compareHist(v_hist1, v_hist2, cv::HISTCMP_BHATTACHARYYA);

    // HS is a stronger indicator of scene identity; V tolerates lighting variation.
    return clamp01(0.70 * (1.0 - hs_bhattacharyya) + 0.30 * (1.0 - v_bhattacharyya));
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
