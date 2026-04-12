#include <opencv2/core/utils/logger.hpp>

#include "ai_detector.hpp"
#include "context_analyzer.hpp"
#include "derivation_analyzer.hpp"
#include "utils.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace
{
std::string yesNo(bool value)
{
    return value ? "YES" : "NO";
}

double averageAiScore(const std::vector<AiDetectionResult>& results)
{
    if (results.empty())
    {
        return 0.0;
    }

    const double total = std::accumulate(
        results.begin(),
        results.end(),
        0.0,
        [](double sum, const AiDetectionResult& result)
        {
            return sum + result.score;
        });

    return total / static_cast<double>(results.size());
}
}  // namespace

int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [output_dir]\n";
        return 1;
    }

    try
    {
        const std::string image_path1 = argv[1];
        const std::string image_path2 = argv[2];
        const std::string output_directory = argc == 4
                                                 ? argv[3]
                                                 : (std::filesystem::path("../data") / "out").string();

        const cv::Mat image1 = utils::loadColorImage(image_path1);
        const cv::Mat image2 = utils::loadColorImage(image_path2);

        ContextAnalyzer context_analyzer;
        DerivationAnalyzer derivation_analyzer;
        AiDetector ai_detector;

        const ContextAnalysisResult context = context_analyzer.analyze(image1, image2);
        const DerivationAnalysisResult derivation = derivation_analyzer.analyze(image1, image2, output_directory);
        const AiDetectionResult ai1 = ai_detector.analyze(image1);
        const AiDetectionResult ai2 = ai_detector.analyze(image2);
        const std::vector<AiDetectionResult> ai_results = {ai1, ai2};
        const double average_ai_score = averageAiScore(ai_results);

        std::cout << "[Context Similarity]\n";
        std::cout << "Score: " << utils::formatScore(context.score)
                  << " -> Same scene: " << yesNo(context.same_context) << '\n';
        std::cout << "Details: " << context.summary << '\n';
        std::cout << "Keypoints: " << context.keypoints_image1
                  << " / " << context.keypoints_image2
                  << ", good matches: " << context.good_matches
                  << ", inliers: " << context.homography_inliers << "\n\n";

        std::cout << "[Derivation Analysis]\n";
        std::cout << "Score: " << utils::formatScore(derivation.score)
                  << " -> Image 2 from Image 1: " << yesNo(derivation.likely_derived) << '\n';
        std::cout << "Details: " << derivation.summary << "\n\n";
        if (!derivation.overlay_path.empty())
        {
            std::cout << "Artifacts saved to:\n";
            std::cout << "  aligned: " << derivation.aligned_image_path << '\n';
            std::cout << "  mask:    " << derivation.change_mask_path << '\n';
            std::cout << "  overlay: " << derivation.overlay_path << "\n\n";
        }

        std::cout << "[AI Generation Suspicion]\n";
        std::cout << "Score: " << utils::formatScore(average_ai_score)
                  << " -> " << AiDetector::suspicionLabel(average_ai_score) << '\n';
        for (std::size_t index = 0; index < ai_results.size(); ++index)
        {
            const AiDetectionResult& result = ai_results[index];
            std::cout << "Image " << (index + 1) << ": "
                      << utils::formatScore(result.score)
                      << " -> " << result.summary << '\n';
        }
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
