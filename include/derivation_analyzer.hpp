#pragma once

#include <opencv2/core.hpp>

#include <string>

struct DerivationAnalysisResult
{
    double score = 0.0;
    bool likely_derived = false;
    int alignment_inliers = 0;
    double alignment_inlier_ratio = 0.0;
    int changed_regions = 0;
    double changed_area_ratio = 0.0;
    double unchanged_similarity = 0.0;
    double cleanup_consistency = 0.0;
    std::string confidence;
    std::string aligned_image_path;
    std::string change_mask_path;
    std::string overlay_path;
    std::string homography_matches_path;
    std::string summary;
};

class DerivationAnalyzer
{
public:
    DerivationAnalysisResult analyze(const cv::Mat& image1,
                                     const cv::Mat& image2,
                                     const std::string& output_directory = "") const;
};
