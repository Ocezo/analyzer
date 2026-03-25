#pragma once

#include <opencv2/core.hpp>

#include <string>

struct DerivationAnalysisResult
{
    double score = 0.0;
    bool likely_derived = false;
    std::string summary;
};

class DerivationAnalyzer
{
public:
    DerivationAnalysisResult analyze(const cv::Mat& image1, const cv::Mat& image2) const;
};
