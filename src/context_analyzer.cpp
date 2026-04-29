#include "context_analyzer.hpp"

#include "utils.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <sstream>

namespace
{
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
    return analyze(utils::computeFeatureAlignment(image1, image2));
}

ContextAnalysisResult ContextAnalyzer::analyze(const utils::FeatureMatchData& fmd) const
{
    ContextAnalysisResult result;

    if (fmd.resized1.empty() || fmd.resized2.empty())
    {
        result.summary = "At least one input image is empty.";
        return result;
    }

    result.keypoints_image1 = static_cast<int>(fmd.keypoints1.size());
    result.keypoints_image2 = static_cast<int>(fmd.keypoints2.size());
    result.color_similarity = utils::compareColorHistograms(fmd.resized1, fmd.resized2);
    result.texture_similarity = utils::computeTextureSimilarity(fmd.gray1, fmd.gray2);
    result.structural_similarity = utils::computeStructuralSimilarity(fmd.gray1, fmd.gray2);

    if (!fmd.descriptors_available)
    {
        result.score = utils::clamp01(0.45 * result.structural_similarity +
                                      0.35 * result.color_similarity +
                                      0.20 * result.texture_similarity);
        result.same_context = result.score >= kDecisionThreshold;
        result.summary = "Not enough local features were detected to verify the scene geometrically.";
        return result;
    }

    result.good_matches = static_cast<int>(fmd.good_matches.size());
    result.homography_inliers =
        static_cast<int>(std::count(fmd.inlier_mask.begin(), fmd.inlier_mask.end(), 1));
    if (!fmd.good_matches.empty())
    {
        result.inlier_ratio = static_cast<double>(result.homography_inliers) /
                              static_cast<double>(fmd.good_matches.size());
    }

    const double match_score = computeMatchScore(
        fmd.resized1, fmd.resized2, fmd.keypoints1, fmd.keypoints2, fmd.good_matches, fmd.inlier_mask);
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
