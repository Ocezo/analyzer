#pragma once

#include <opencv2/core.hpp>

#include <string>

struct AiDetectionResult
{
    double score = 0.0;
    double noise_score = 0.0;
    double jpeg_score = 0.0;
    double frequency_score = 0.0;
    double edge_uniformity_score = 0.0;
    std::string summary;
};

class AiDetector
{
public:
    AiDetectionResult analyze(const cv::Mat& image) const;
    static std::string suspicionLabel(double score);
};
