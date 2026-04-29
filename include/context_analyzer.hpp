#pragma once

#include "utils.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct ContextAnalysisResult
{
    double score = 0.0;
    bool same_context = false;
    int keypoints_image1 = 0;
    int keypoints_image2 = 0;
    int good_matches = 0;
    int homography_inliers = 0;
    double inlier_ratio = 0.0;
    double color_similarity = 0.0;
    double texture_similarity = 0.0;
    double structural_similarity = 0.0;
    std::string confidence;
    std::string summary;
};

class ContextAnalyzer
{
public:
    // Convenience overload: computes feature alignment internally.
    ContextAnalysisResult analyze(const cv::Mat& image1, const cv::Mat& image2) const;
    // Primary overload: uses pre-computed feature alignment (avoids redundant computation).
    ContextAnalysisResult analyze(const utils::FeatureMatchData& fmd) const;

private:
    static double computeMatchScore(const cv::Mat& image1,
                                    const cv::Mat& image2,
                                    const std::vector<cv::KeyPoint>& keypoints1,
                                    const std::vector<cv::KeyPoint>& keypoints2,
                                    const std::vector<cv::DMatch>& matches,
                                    const std::vector<unsigned char>& inlier_mask);
};
