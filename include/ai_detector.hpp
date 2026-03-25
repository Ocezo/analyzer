#pragma once

#include <opencv2/core.hpp>

#include <string>

struct AiDetectionResult
{
    double score = 0.0;
    std::string summary;
};

class AiDetector
{
public:
    AiDetectionResult analyze(const cv::Mat& image) const;
};
