#include "ai_detector.hpp"
#include "context_analyzer.hpp"
#include "derivation_analyzer.hpp"
#include "utils.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace
{
std::string yesNo(bool value)
{
    return value ? "YES" : "NO";
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2>\n";
        return 1;
    }

    try
    {
        const std::string image_path1 = argv[1];
        const std::string image_path2 = argv[2];

        const cv::Mat image1 = utils::loadColorImage(image_path1);
        const cv::Mat image2 = utils::loadColorImage(image_path2);

        ContextAnalyzer context_analyzer;
        DerivationAnalyzer derivation_analyzer;
        AiDetector ai_detector;

        const ContextAnalysisResult context = context_analyzer.analyze(image1, image2);
        const DerivationAnalysisResult derivation = derivation_analyzer.analyze(image1, image2);
        const AiDetectionResult ai1 = ai_detector.analyze(image1);
        const AiDetectionResult ai2 = ai_detector.analyze(image2);

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
                  << " -> Image 2 likely derived from Image 1: " << yesNo(derivation.likely_derived) << '\n';
        std::cout << "Details: " << derivation.summary << "\n\n";

        std::cout << "[AI Generation Suspicion]\n";
        std::cout << "Image 1: " << utils::formatScore(ai1.score) << " -> " << ai1.summary << '\n';
        std::cout << "Image 2: " << utils::formatScore(ai2.score) << " -> " << ai2.summary << '\n';
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
