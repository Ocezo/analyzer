#include "derivation_analyzer.hpp"

DerivationAnalysisResult DerivationAnalyzer::analyze(const cv::Mat&, const cv::Mat&) const
{
    return {
        0.0,
        false,
        "Derivation analysis is not implemented yet."
    };
}
