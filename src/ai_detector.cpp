#include "ai_detector.hpp"

#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <vector>

namespace
{
constexpr int kAnalysisSize = 256;
constexpr int kPatchSize = 32;

cv::Mat prepareGrayImage(const cv::Mat& image)
{
    cv::Mat resized = utils::resizeToMaxSide(image, 1024);
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(kAnalysisSize, kAnalysisSize), 0.0, 0.0, cv::INTER_AREA);
    return gray;
}

cv::Mat prepareGrayForJpeg(const cv::Mat& image)
{
    // Keep original resolution (max 1024) so that JPEG 8x8 block boundaries are preserved.
    cv::Mat resized = utils::resizeToMaxSide(image, 1024);
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

double computeNoiseSuspicion(const cv::Mat& gray)
{
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0.0);

    cv::Mat residual;
    cv::absdiff(gray, blurred, residual);

    std::vector<double> patch_stddevs;
    std::vector<double> patch_means;

    for (int y = 0; y + kPatchSize <= residual.rows; y += kPatchSize)
    {
        for (int x = 0; x + kPatchSize <= residual.cols; x += kPatchSize)
        {
            const cv::Rect roi(x, y, kPatchSize, kPatchSize);
            cv::Scalar mean;
            cv::Scalar stddev;
            cv::meanStdDev(residual(roi), mean, stddev);
            patch_means.push_back(mean[0]);
            patch_stddevs.push_back(stddev[0]);
        }
    }

    if (patch_means.empty())
    {
        return 0.0;
    }

    const double mean_noise = std::accumulate(patch_means.begin(), patch_means.end(), 0.0) /
                              static_cast<double>(patch_means.size());
    const double mean_stddev = std::accumulate(patch_stddevs.begin(), patch_stddevs.end(), 0.0) /
                               static_cast<double>(patch_stddevs.size());

    double patch_variation = 0.0;
    for (double value : patch_means)
    {
        const double delta = value - mean_noise;
        patch_variation += delta * delta;
    }
    patch_variation = std::sqrt(patch_variation / static_cast<double>(patch_means.size()));

    const double overly_smooth = utils::clamp01((6.5 - mean_noise) / 6.5);
    const double low_variation = utils::clamp01((1.8 - patch_variation) / 1.8);
    const double low_microcontrast = utils::clamp01((7.0 - mean_stddev) / 7.0);

    return utils::clamp01(0.40 * overly_smooth +
                          0.35 * low_variation +
                          0.25 * low_microcontrast);
}

double computeJpegSuspicion(const cv::Mat& gray)
{
    double boundary_sum = 0.0;
    double interior_sum = 0.0;
    int boundary_count = 0;
    int interior_count = 0;

    for (int y = 0; y < gray.rows; ++y)
    {
        for (int x = 1; x < gray.cols; ++x)
        {
            const double diff = std::abs(gray.at<unsigned char>(y, x) - gray.at<unsigned char>(y, x - 1));
            if (x % 8 == 0)
            {
                boundary_sum += diff;
                ++boundary_count;
            }
            else
            {
                interior_sum += diff;
                ++interior_count;
            }
        }
    }

    for (int y = 1; y < gray.rows; ++y)
    {
        for (int x = 0; x < gray.cols; ++x)
        {
            const double diff = std::abs(gray.at<unsigned char>(y, x) - gray.at<unsigned char>(y - 1, x));
            if (y % 8 == 0)
            {
                boundary_sum += diff;
                ++boundary_count;
            }
            else
            {
                interior_sum += diff;
                ++interior_count;
            }
        }
    }

    if (boundary_count == 0 || interior_count == 0)
    {
        return 0.0;
    }

    const double boundary_mean = boundary_sum / static_cast<double>(boundary_count);
    const double interior_mean = interior_sum / static_cast<double>(interior_count);
    const double ratio = boundary_mean / std::max(interior_mean, 1.0);

    return utils::clamp01((ratio - 1.05) / 0.35);
}

double computeFrequencySuspicion(const cv::Mat& gray)
{
    cv::Mat float_image;
    gray.convertTo(float_image, CV_32F, 1.0 / 255.0);

    cv::Mat complex_image;
    cv::dft(float_image, complex_image, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes;
    cv::split(complex_image, planes);

    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    magnitude += 1.0f;
    cv::log(magnitude, magnitude);

    const int cx = magnitude.cols / 2;
    const int cy = magnitude.rows / 2;
    cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    double ring_sum = 0.0;
    double ring_sq_sum = 0.0;
    int ring_count = 0;
    double peak_value = 0.0;

    for (int y = 0; y < magnitude.rows; ++y)
    {
        for (int x = 0; x < magnitude.cols; ++x)
        {
            const double dx = static_cast<double>(x - cx) / static_cast<double>(cx);
            const double dy = static_cast<double>(y - cy) / static_cast<double>(cy);
            const double radius = std::sqrt(dx * dx + dy * dy);
            if (radius < 0.18 || radius > 0.75)
            {
                continue;
            }

            const double value = magnitude.at<float>(y, x);
            ring_sum += value;
            ring_sq_sum += value * value;
            peak_value = std::max(peak_value, value);
            ++ring_count;
        }
    }

    if (ring_count == 0)
    {
        return 0.0;
    }

    const double mean = ring_sum / static_cast<double>(ring_count);
    const double variance = std::max(0.0, ring_sq_sum / static_cast<double>(ring_count) - mean * mean);
    const double stddev = std::sqrt(variance);
    const double peak_zscore = (peak_value - mean) / std::max(stddev, 1e-6);

    return utils::clamp01((peak_zscore - 3.5) / 4.5);
}

// AI-generated images tend to have unnaturally uniform edge sharpness across the image:
// real photos show high local variation (sharp foreground, blurry background, noise).
double computeEdgeUniformitySuspicion(const cv::Mat& gray)
{
    cv::Mat sobelx;
    cv::Mat sobely;
    cv::Sobel(gray, sobelx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sobely, CV_32F, 0, 1, 3);

    cv::Mat gradient_magnitude;
    cv::magnitude(sobelx, sobely, gradient_magnitude);

    std::vector<double> patch_energies;
    for (int y = 0; y + kPatchSize <= gradient_magnitude.rows; y += kPatchSize)
    {
        for (int x = 0; x + kPatchSize <= gradient_magnitude.cols; x += kPatchSize)
        {
            const cv::Rect roi(x, y, kPatchSize, kPatchSize);
            const cv::Scalar m = cv::mean(gradient_magnitude(roi));
            patch_energies.push_back(m[0]);
        }
    }

    if (patch_energies.size() < 4)
    {
        return 0.0;
    }

    const double mean_energy = std::accumulate(patch_energies.begin(), patch_energies.end(), 0.0) /
                               static_cast<double>(patch_energies.size());
    double sq_sum = 0.0;
    for (double v : patch_energies)
    {
        sq_sum += (v - mean_energy) * (v - mean_energy);
    }
    const double energy_stddev = std::sqrt(sq_sum / static_cast<double>(patch_energies.size()));
    // Low coefficient of variation → unnaturally uniform edge distribution → suspicious.
    // Real photos naturally vary (depth-of-field, local textures): CV > 0.35 is normal.
    // AI images often sit below 0.15.
    const double cv_ratio = energy_stddev / std::max(mean_energy, 1.0);
    return utils::clamp01((0.25 - cv_ratio) / 0.25);
}

}  // namespace

std::string AiDetector::suspicionLabel(double score)
{
    if (score >= 0.70)
    {
        return "High suspicion";
    }
    if (score >= 0.40)
    {
        return "Moderate suspicion";
    }
    return "Low suspicion";
}

AiDetectionResult AiDetector::analyze(const cv::Mat& image) const
{
    AiDetectionResult result;

    if (image.empty())
    {
        result.summary = "Empty image.";
        return result;
    }

    const cv::Mat gray = prepareGrayImage(image);
    const cv::Mat gray_full = prepareGrayForJpeg(image);

    result.noise_score = computeNoiseSuspicion(gray);
    result.jpeg_score = computeJpegSuspicion(gray_full);
    result.frequency_score = computeFrequencySuspicion(gray);
    // Use the larger (1024px) image for edge analysis: more patches → better statistics.
    result.edge_uniformity_score = computeEdgeUniformitySuspicion(gray_full);
    result.score = utils::clamp01(0.35 * result.noise_score +
                                  0.25 * result.frequency_score +
                                  0.25 * result.jpeg_score +
                                  0.15 * result.edge_uniformity_score);

    std::ostringstream summary;
    summary << AiDetector::suspicionLabel(result.score)
            << " (noise: " << utils::formatScore(result.noise_score)
            << ", jpeg: " << utils::formatScore(result.jpeg_score)
            << ", frequency: " << utils::formatScore(result.frequency_score)
            << ", edge_uniformity: " << utils::formatScore(result.edge_uniformity_score) << ")";
    result.summary = summary.str();

    return result;
}
