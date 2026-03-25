#include "derivation_analyzer.hpp"

#include "utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

namespace
{
constexpr int kMaxSide = 1400;
constexpr int kMaxFeatures = 2500;
constexpr float kRatioTest = 0.78f;
constexpr double kHomographyRansacThreshold = 3.0;
constexpr double kDecisionThreshold = 0.55;

cv::Mat computeChangeMask(const cv::Mat& aligned1, const cv::Mat& image2)
{
    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(aligned1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    cv::Mat blur1;
    cv::Mat blur2;
    cv::GaussianBlur(gray1, blur1, cv::Size(7, 7), 0.0);
    cv::GaussianBlur(gray2, blur2, cv::Size(7, 7), 0.0);

    cv::Mat difference;
    cv::absdiff(blur1, blur2, difference);

    cv::Mat binary_mask;
    cv::threshold(difference, binary_mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

    const int border_x = std::max(8, binary_mask.cols / 50);
    const int border_y = std::max(8, binary_mask.rows / 50);
    binary_mask(cv::Rect(0, 0, binary_mask.cols, border_y)).setTo(0);
    binary_mask(cv::Rect(0, binary_mask.rows - border_y, binary_mask.cols, border_y)).setTo(0);
    binary_mask(cv::Rect(0, 0, border_x, binary_mask.rows)).setTo(0);
    binary_mask(cv::Rect(binary_mask.cols - border_x, 0, border_x, binary_mask.rows)).setTo(0);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered_mask = cv::Mat::zeros(binary_mask.size(), CV_8U);
    const int min_component_area = std::max(250, binary_mask.total() > 0 ? static_cast<int>(binary_mask.total() / 1200) : 250);
    for (int label = 1; label < component_count; ++label)
    {
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= min_component_area)
        {
            filtered_mask.setTo(255, labels == label);
        }
    }

    return filtered_mask;
}

int countChangedRegions(const cv::Mat& change_mask)
{
    cv::Mat labels;
    return std::max(0, cv::connectedComponents(change_mask, labels, 8, CV_32S) - 1);
}

double computeCleanupConsistency(const cv::Mat& aligned1, const cv::Mat& image2, const cv::Mat& change_mask)
{
    if (cv::countNonZero(change_mask) == 0)
    {
        return 0.0;
    }

    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(aligned1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    cv::Scalar mean1;
    cv::Scalar stddev1;
    cv::Scalar mean2;
    cv::Scalar stddev2;
    cv::meanStdDev(gray1, mean1, stddev1, change_mask);
    cv::meanStdDev(gray2, mean2, stddev2, change_mask);

    cv::Mat laplacian1;
    cv::Mat laplacian2;
    cv::Laplacian(gray1, laplacian1, CV_32F, 3);
    cv::Laplacian(gray2, laplacian2, CV_32F, 3);
    cv::Mat abs_laplacian1 = cv::abs(laplacian1);
    cv::Mat abs_laplacian2 = cv::abs(laplacian2);

    const cv::Scalar mean_detail1 = cv::mean(abs_laplacian1, change_mask);
    const cv::Scalar mean_detail2 = cv::mean(abs_laplacian2, change_mask);

    const double detail_delta = mean_detail1[0] - mean_detail2[0];
    const double detail_scale = std::max({mean_detail1[0], mean_detail2[0], 1.0});
    const double variance_delta = stddev1[0] - stddev2[0];
    const double variance_scale = std::max({stddev1[0], stddev2[0], 1.0});
    const double brightness_gap = std::abs(mean1[0] - mean2[0]);
    const double brightness_score = utils::clamp01(1.0 - brightness_gap / 64.0);

    return utils::clamp01(0.50 * (0.5 + 0.5 * detail_delta / detail_scale) +
                          0.35 * (0.5 + 0.5 * variance_delta / variance_scale) +
                          0.15 * brightness_score);
}

double computeUnchangedSimilarity(const cv::Mat& aligned1, const cv::Mat& image2, const cv::Mat& change_mask)
{
    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(aligned1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    cv::Mat unchanged_mask;
    cv::bitwise_not(change_mask, unchanged_mask);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::erode(unchanged_mask, unchanged_mask, kernel);

    if (cv::countNonZero(unchanged_mask) == 0)
    {
        return 0.0;
    }

    cv::Mat difference;
    cv::absdiff(gray1, gray2, difference);
    const cv::Scalar mean_difference = cv::mean(difference, unchanged_mask);
    return utils::clamp01(1.0 - mean_difference[0] / 255.0);
}
}  // namespace

DerivationAnalysisResult DerivationAnalyzer::analyze(const cv::Mat& image1, const cv::Mat& image2) const
{
    DerivationAnalysisResult result;

    if (image1.empty() || image2.empty())
    {
        result.summary = "At least one input image is empty.";
        return result;
    }

    const cv::Mat resized1 = utils::resizeToMaxSide(image1, kMaxSide);
    const cv::Mat resized2 = utils::resizeToMaxSide(image2, kMaxSide);

    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(resized1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(resized2, gray2, cv::COLOR_BGR2GRAY);

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

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    detector->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    if (descriptors1.empty() || descriptors2.empty())
    {
        result.summary = "Not enough local features were detected to align the images.";
        return result;
    }

    cv::BFMatcher matcher(matcher_norm);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto& pair : knn_matches)
    {
        if (pair.size() >= 2 && pair[0].distance < kRatioTest * pair[1].distance)
        {
            good_matches.push_back(pair[0]);
        }
    }

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (const auto& match : good_matches)
    {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::vector<unsigned char> inlier_mask;
    cv::Mat homography;
    if (points1.size() >= 4)
    {
        homography = cv::findHomography(points1, points2, cv::RANSAC, kHomographyRansacThreshold, inlier_mask);
    }

    if (homography.empty())
    {
        result.summary = "Unable to estimate a stable alignment between the two images.";
        return result;
    }

    result.alignment_inliers = static_cast<int>(std::count(inlier_mask.begin(), inlier_mask.end(), 1));
    if (!good_matches.empty())
    {
        result.alignment_inlier_ratio = static_cast<double>(result.alignment_inliers) /
                                        static_cast<double>(good_matches.size());
    }

    cv::Mat aligned1;
    cv::warpPerspective(resized1, aligned1, homography, resized2.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    const cv::Mat change_mask = computeChangeMask(aligned1, resized2);
    result.changed_regions = countChangedRegions(change_mask);
    result.changed_area_ratio = static_cast<double>(cv::countNonZero(change_mask)) /
                                static_cast<double>(change_mask.total());

    result.unchanged_similarity = computeUnchangedSimilarity(aligned1, resized2, change_mask);
    result.cleanup_consistency = computeCleanupConsistency(aligned1, resized2, change_mask);

    const double alignment_score = utils::clamp01(0.60 * result.alignment_inlier_ratio +
                                                  0.40 * std::min(1.0, result.alignment_inliers / 25.0));
    const double change_score = utils::clamp01(result.changed_area_ratio / 0.12);
    const double preservation_score = utils::clamp01((result.unchanged_similarity - 0.45) / 0.40);

    result.score = utils::clamp01(0.30 * alignment_score +
                                  0.25 * change_score +
                                  0.25 * result.cleanup_consistency +
                                  0.20 * preservation_score);

    result.likely_derived = result.score >= kDecisionThreshold &&
                            result.alignment_inliers >= 6 &&
                            result.changed_regions >= 1 &&
                            result.changed_area_ratio >= 0.005 &&
                            result.unchanged_similarity >= 0.75 &&
                            result.cleanup_consistency >= 0.40;

    std::ostringstream summary;
    summary << "Alignment inliers: " << result.alignment_inliers
            << " (" << utils::formatScore(result.alignment_inlier_ratio) << ")"
            << ", changed regions: " << result.changed_regions
            << ", changed area ratio: " << utils::formatScore(result.changed_area_ratio)
            << ", unchanged similarity: " << utils::formatScore(result.unchanged_similarity)
            << ", cleanup consistency: " << utils::formatScore(result.cleanup_consistency);
    result.summary = summary.str();

    return result;
}
