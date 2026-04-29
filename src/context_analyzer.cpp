#include "context_analyzer.hpp"

#include "utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <sstream>

namespace
{
constexpr int kMaxSide = 1400;
constexpr int kMaxFeatures = 2500;
constexpr float kRatioTest = 0.75f;
constexpr double kHomographyRansacThreshold = 5.0;
constexpr double kDecisionThreshold = 0.50;
}  // namespace

ContextAnalysisResult ContextAnalyzer::analyze(const cv::Mat& image1, const cv::Mat& image2) const
{
    ContextAnalysisResult result;

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

    result.keypoints_image1 = static_cast<int>(keypoints1.size());
    result.keypoints_image2 = static_cast<int>(keypoints2.size());
    result.color_similarity = utils::compareColorHistograms(resized1, resized2);
    result.texture_similarity = utils::computeTextureSimilarity(gray1, gray2);
    result.structural_similarity = utils::computeStructuralSimilarity(gray1, gray2);

    if (descriptors1.empty() || descriptors2.empty())
    {
        result.score = utils::clamp01(0.45 * result.structural_similarity +
                                      0.35 * result.color_similarity +
                                      0.20 * result.texture_similarity);
        result.same_context = result.score >= kDecisionThreshold;
        result.summary = "Not enough local features were detected to verify the scene geometrically.";
        return result;
    }

    cv::BFMatcher matcher(matcher_norm);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());

    for (const auto& pair : knn_matches)
    {
        if (pair.size() < 2)
        {
            continue;
        }

        if (pair[0].distance < kRatioTest * pair[1].distance)
        {
            good_matches.push_back(pair[0]);
        }
    }

    result.good_matches = static_cast<int>(good_matches.size());

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    points1.reserve(good_matches.size());
    points2.reserve(good_matches.size());

    for (const auto& match : good_matches)
    {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::vector<unsigned char> inlier_mask;
    if (points1.size() >= 4)
    {
        cv::findHomography(points1, points2, cv::RANSAC, kHomographyRansacThreshold, inlier_mask);
    }

    result.homography_inliers = static_cast<int>(std::count(inlier_mask.begin(), inlier_mask.end(), 1));
    if (!good_matches.empty())
    {
        result.inlier_ratio = static_cast<double>(result.homography_inliers) / static_cast<double>(good_matches.size());
    }

    const double match_score = computeMatchScore(resized1, resized2, keypoints1, keypoints2, good_matches, inlier_mask);
    result.score = utils::clamp01(0.35 * match_score +
                                  0.30 * result.structural_similarity +
                                  0.20 * result.color_similarity +
                                  0.15 * result.texture_similarity);
    // Strong geometric evidence OR consistent global appearance (covers scenes where feature
    // matching is unreliable: plants, water, repetitive textures).
    result.same_context = result.score >= kDecisionThreshold &&
                          (result.structural_similarity >= 0.67 ||
                           (result.homography_inliers >= 12 && result.inlier_ratio >= 0.25) ||
                           (result.color_similarity >= 0.70 && result.texture_similarity >= 0.80));

    if (result.homography_inliers >= 25 && result.score >= 0.70)
    {
        result.confidence = "High";
    }
    else if (result.homography_inliers >= 8 || result.structural_similarity >= 0.70)
    {
        result.confidence = "Medium";
    }
    else
    {
        result.confidence = "Low";
    }

    std::ostringstream summary;
    summary << "Feature matches: " << result.good_matches
            << ", homography inliers: " << result.homography_inliers
            << ", color similarity: " << utils::formatScore(result.color_similarity)
            << ", texture similarity: " << utils::formatScore(result.texture_similarity)
            << ", structural similarity: " << utils::formatScore(result.structural_similarity);
    result.summary = summary.str();

    return result;
}

double ContextAnalyzer::computeMatchScore(const cv::Mat& image1,
                                          const cv::Mat& image2,
                                          const std::vector<cv::KeyPoint>& keypoints1,
                                          const std::vector<cv::KeyPoint>& keypoints2,
                                          const std::vector<cv::DMatch>& matches,
                                          const std::vector<unsigned char>& inlier_mask)
{
    if (matches.empty())
    {
        return 0.0;
    }

    const double normalized_match_count = std::min(
        1.0,
        static_cast<double>(matches.size()) /
            std::max(40.0, 0.03 * static_cast<double>(std::min(keypoints1.size(), keypoints2.size()))));

    double inlier_ratio = 0.0;
    if (!inlier_mask.empty())
    {
        const auto inliers = static_cast<double>(std::count(inlier_mask.begin(), inlier_mask.end(), 1));
        inlier_ratio = inliers / static_cast<double>(matches.size());
    }

    const double area_scale = static_cast<double>(std::min(image1.cols * image1.rows, image2.cols * image2.rows));
    const double density = std::min(1.0, static_cast<double>(matches.size()) / std::max(1.0, area_scale / 25000.0));

    return utils::clamp01(0.45 * normalized_match_count + 0.40 * inlier_ratio + 0.15 * density);
}
