#include "ai_detector.hpp"

AiDetectionResult AiDetector::analyze(const cv::Mat&) const
{
    return {
        0.0,
        "AI suspicion analysis is not implemented yet."
    };
}
