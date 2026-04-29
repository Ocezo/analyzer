#include "derivation_analyzer.hpp"

#include "utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <vector>

namespace
{
constexpr double kDecisionThreshold = 0.55;

void drawCross(cv::Mat& img, cv::Point2f pt, const cv::Scalar& color, int arm = 5)
{
    const cv::Point p(cvRound(pt.x), cvRound(pt.y));
    cv::line(img, {p.x - arm, p.y}, {p.x + arm, p.y}, color, 1, cv::LINE_AA);
    cv::line(img, {p.x, p.y - arm}, {p.x, p.y + arm}, color, 1, cv::LINE_AA);
}

cv::Mat computeChangeMask(const cv::Mat& aligned1, const cv::Mat& image2)
{
    // Work in Lab so that perceptual color differences (not just luminance) are captured.
    // This detects objects removed even if they have similar brightness to the background.
    cv::Mat lab1;
    cv::Mat lab2;
    cv::cvtColor(aligned1, lab1, cv::COLOR_BGR2Lab);
    cv::cvtColor(image2, lab2, cv::COLOR_BGR2Lab);

    cv::Mat blur1;
    cv::Mat blur2;
    cv::GaussianBlur(lab1, blur1, cv::Size(7, 7), 0.0);
    cv::GaussianBlur(lab2, blur2, cv::Size(7, 7), 0.0);

    // Per-channel absolute difference then merge into a single perceptual distance map.
    std::vector<cv::Mat> diff_channels(3);
    for (int c = 0; c < 3; ++c)
    {
        cv::Mat ch1;
        cv::Mat ch2;
        cv::extractChannel(blur1, ch1, c);
        cv::extractChannel(blur2, ch2, c);
        cv::absdiff(ch1, ch2, diff_channels[c]);
        diff_channels[c].convertTo(diff_channels[c], CV_32F);
    }

    // deltaE approximation: weight L* less than a*/b* to tolerate illumination variation.
    cv::Mat difference;
    cv::Mat weighted = 0.5f * diff_channels[0] + 0.85f * diff_channels[1] + 0.85f * diff_channels[2];
    weighted.convertTo(difference, CV_8U, 1.0, 0.0);

    cv::Mat binary_mask;
    cv::threshold(difference, binary_mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

    const int border_x = std::max(8, binary_mask.cols / 50);
    const int border_y = std::max(8, binary_mask.rows / 50);
    binary_mask(cv::Rect(0, 0, binary_mask.cols, border_y)).setTo(0);
    binary_mask(cv::Rect(0, binary_mask.rows - border_y, binary_mask.cols, border_y)).setTo(0);
    binary_mask(cv::Rect(0, 0, border_x, binary_mask.rows)).setTo(0);
    binary_mask(cv::Rect(binary_mask.cols - border_x, 0, border_x, binary_mask.rows)).setTo(0);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered_mask = cv::Mat::zeros(binary_mask.size(), CV_8U);
    // Require larger minimum area (1/600 of image instead of 1/1200) to suppress
    // small wind-induced leaf / texture movement artefacts.
    const int total_px = static_cast<int>(binary_mask.total());
    const int min_component_area = std::max(400, total_px > 0 ? total_px / 600 : 400);
    for (int label = 1; label < component_count; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_component_area)
        {
            continue;
        }
        // Discard very thin/elongated regions (alignment border artefacts).
        const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const double aspect = static_cast<double>(std::max(w, h)) / static_cast<double>(std::max(std::min(w, h), 1));
        if (aspect > 12.0)
        {
            continue;
        }
        filtered_mask.setTo(255, labels == label);
    }

    return filtered_mask;
}

int countChangedRegions(const cv::Mat& change_mask)
{
    cv::Mat labels;
    return std::max(0, cv::connectedComponents(change_mask, labels, 8, CV_32S) - 1);
}

double computeCleanupConsistency(const cv::Mat& aligned1, const cv::Mat& image2, const cv::Mat& change_mask)
{
    if (cv::countNonZero(change_mask) == 0)
    {
        return 0.0;
    }

    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(aligned1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    // (a) Changed regions in image1 should have MORE detail than in image2 (object removed).
    cv::Mat laplacian1;
    cv::Mat laplacian2;
    cv::Laplacian(gray1, laplacian1, CV_32F, 3);
    cv::Laplacian(gray2, laplacian2, CV_32F, 3);
    cv::Mat abs_laplacian1 = cv::abs(laplacian1);
    cv::Mat abs_laplacian2 = cv::abs(laplacian2);

    const cv::Scalar mean_detail1 = cv::mean(abs_laplacian1, change_mask);
    const cv::Scalar mean_detail2 = cv::mean(abs_laplacian2, change_mask);
    const double detail_delta = mean_detail1[0] - mean_detail2[0];
    const double detail_scale = std::max({mean_detail1[0], mean_detail2[0], 1.0});
    const double object_removal_score = utils::clamp01(0.5 + 0.5 * detail_delta / detail_scale);

    // (b) Variance in changed regions should be lower in image2 (background is smoother).
    cv::Scalar m1, sd1, m2, sd2;
    cv::meanStdDev(gray1, m1, sd1, change_mask);
    cv::meanStdDev(gray2, m2, sd2, change_mask);
    const double variance_delta = sd1[0] - sd2[0];
    const double variance_scale = std::max({sd1[0], sd2[0], 1.0});
    const double smoothing_score = utils::clamp01(0.5 + 0.5 * variance_delta / variance_scale);

    // (c) Background-consistency: changed pixels in image2 should resemble the surrounding
    //     unchanged area in image2 (typical signature of inpainting / copy-fill cleanup).
    cv::Mat border_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
    cv::Mat dilated_mask;
    cv::dilate(change_mask, dilated_mask, border_kernel);
    cv::Mat border_mask;
    cv::bitwise_xor(dilated_mask, change_mask, border_mask);

    double bg_similarity = 0.5;
    if (cv::countNonZero(border_mask) > 0)
    {
        const cv::Scalar bg_mean = cv::mean(gray2, border_mask);
        const cv::Scalar fg_mean = cv::mean(gray2, change_mask);
        const double brightness_gap = std::abs(bg_mean[0] - fg_mean[0]);
        bg_similarity = utils::clamp01(1.0 - brightness_gap / 48.0);
    }

    return utils::clamp01(0.45 * object_removal_score +
                          0.30 * smoothing_score +
                          0.25 * bg_similarity);
}

double computeUnchangedSimilarity(const cv::Mat& aligned1, const cv::Mat& image2, const cv::Mat& change_mask)
{
    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(aligned1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    cv::Mat unchanged_mask;
    cv::bitwise_not(change_mask, unchanged_mask);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::erode(unchanged_mask, unchanged_mask, kernel);

    if (cv::countNonZero(unchanged_mask) == 0)
    {
        return 0.0;
    }

    cv::Mat difference;
    cv::absdiff(gray1, gray2, difference);
    const cv::Scalar mean_difference = cv::mean(difference, unchanged_mask);
    return utils::clamp01(1.0 - mean_difference[0] / 255.0);
}

void exportArtifacts(const cv::Mat& resized1,
                     const cv::Mat& aligned1,
                     const cv::Mat& image2,
                     const cv::Mat& change_mask,
                     const std::vector<cv::KeyPoint>& keypoints1,
                     const std::vector<cv::KeyPoint>& keypoints2,
                     const std::vector<cv::DMatch>& good_matches,
                     const std::vector<unsigned char>& inlier_mask,
                     const std::string& output_directory,
                     DerivationAnalysisResult& result)
{
    if (output_directory.empty())
    {
        return;
    }

    std::filesystem::create_directories(output_directory);

    const std::filesystem::path base(output_directory);
    const std::filesystem::path aligned_path = base / "aligned_image1_to_image2.jpg";
    const std::filesystem::path mask_path = base / "derivation_change_mask.png";
    const std::filesystem::path overlay_path = base / "derivation_overlay.jpg";
    const std::filesystem::path matches_path = base / "homography_matches.jpg";

    cv::Mat overlay = image2.clone();
    cv::Mat red_overlay(image2.size(), image2.type(), cv::Scalar(0, 0, 255));
    red_overlay.copyTo(overlay, change_mask);
    cv::addWeighted(overlay, 0.35, image2, 0.65, 0.0, overlay);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(change_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 255), 2);

    cv::imwrite(aligned_path.string(), aligned1);
    cv::imwrite(mask_path.string(), change_mask);
    cv::imwrite(overlay_path.string(), overlay);

    result.aligned_image_path = aligned_path.string();
    result.change_mask_path = mask_path.string();
    result.overlay_path = overlay_path.string();

    // --- Homography matches visualization ---
    cv::Mat canvas;
    cv::hconcat(resized1, image2, canvas);
    const int offset = resized1.cols;

    // Green crosses: all detected keypoints
    for (const auto& kp : keypoints1)
        drawCross(canvas, kp.pt, cv::Scalar(0, 200, 0));
    for (const auto& kp : keypoints2)
        drawCross(canvas, {kp.pt.x + static_cast<float>(offset), kp.pt.y}, cv::Scalar(0, 200, 0));

    // White lines then red crosses: RANSAC inliers
    for (size_t i = 0; i < good_matches.size(); ++i)
    {
        if (i >= inlier_mask.size() || !inlier_mask[i])
            continue;
        const cv::Point2f pt1 = keypoints1[good_matches[i].queryIdx].pt;
        const cv::Point2f pt2 = {keypoints2[good_matches[i].trainIdx].pt.x + static_cast<float>(offset),
                                  keypoints2[good_matches[i].trainIdx].pt.y};
        cv::line(canvas,
                 cv::Point(cvRound(pt1.x), cvRound(pt1.y)),
                 cv::Point(cvRound(pt2.x), cvRound(pt2.y)),
                 cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        drawCross(canvas, pt1, cv::Scalar(0, 0, 255));
        drawCross(canvas, pt2, cv::Scalar(0, 0, 255));
    }

    cv::imwrite(matches_path.string(), canvas);
    result.homography_matches_path = matches_path.string();
}
}  // namespace

DerivationAnalysisResult DerivationAnalyzer::analyze(const cv::Mat& image1,
                                                     const cv::Mat& image2,
                                                     const std::string& output_directory) const
{
    DerivationAnalysisResult result;
    if (image1.empty() || image2.empty())
    {
        result.summary = "At least one input image is empty.";
        return result;
    }
    return analyze(utils::computeFeatureAlignment(image1, image2), output_directory);
}

DerivationAnalysisResult DerivationAnalyzer::analyze(const utils::FeatureMatchData& fmd,
                                                     const std::string& output_directory) const
{
    DerivationAnalysisResult result;

    if (fmd.resized1.empty() || fmd.resized2.empty())
    {
        result.summary = "At least one input image is empty.";
        return result;
    }

    if (!fmd.descriptors_available)
    {
        result.summary = "Not enough local features were detected to align the images.";
        return result;
    }

    if (fmd.homography.empty())
    {
        result.summary = "Unable to estimate a stable alignment between the two images.";
        return result;
    }

    result.alignment_inliers =
        static_cast<int>(std::count(fmd.inlier_mask.begin(), fmd.inlier_mask.end(), 1));
    if (!fmd.good_matches.empty())
    {
        result.alignment_inlier_ratio = static_cast<double>(result.alignment_inliers) /
                                        static_cast<double>(fmd.good_matches.size());
    }

    cv::Mat aligned1;
    cv::warpPerspective(fmd.resized1, aligned1, fmd.homography, fmd.resized2.size(),
                        cv::INTER_LINEAR, cv::BORDER_REFLECT);

    const cv::Mat change_mask = computeChangeMask(aligned1, fmd.resized2);
    result.changed_regions = countChangedRegions(change_mask);
    result.changed_area_ratio = static_cast<double>(cv::countNonZero(change_mask)) /
                                static_cast<double>(change_mask.total());

    result.unchanged_similarity = computeUnchangedSimilarity(aligned1, fmd.resized2, change_mask);
    result.cleanup_consistency = computeCleanupConsistency(aligned1, fmd.resized2, change_mask);
    exportArtifacts(fmd.resized1, aligned1, fmd.resized2, change_mask,
                    fmd.keypoints1, fmd.keypoints2, fmd.good_matches, fmd.inlier_mask,
                    output_directory, result);

    const double alignment_score = utils::clamp01(0.60 * result.alignment_inlier_ratio +
                                                  0.40 * std::min(1.0, result.alignment_inliers / 25.0));
    // Bell-curve change score: rises to 1.0 at ~10%, stays flat up to 35%, then
    // declines for very large changes (> 80% likely indicates alignment failure).
    const double area = result.changed_area_ratio;
    const double change_score = area < 0.003
                                    ? 0.0
                                    : area <= 0.35
                                          ? utils::clamp01(area / 0.10)
                                          : utils::clamp01(1.0 - (area - 0.35) / 0.55);
    const double preservation_score = utils::clamp01((result.unchanged_similarity - 0.45) / 0.40);

    result.score = utils::clamp01(0.30 * alignment_score +
                                  0.25 * change_score +
                                  0.25 * result.cleanup_consistency +
                                  0.20 * preservation_score);

    result.likely_derived = result.score >= kDecisionThreshold &&
                            result.alignment_inliers >= 6 &&
                            result.changed_regions >= 1 &&
                            result.changed_area_ratio >= 0.005 &&
                            result.unchanged_similarity >= 0.75 &&
                            result.cleanup_consistency >= 0.40;

    if (result.alignment_inliers >= 30 && result.unchanged_similarity >= 0.82 && result.score >= 0.70)
    {
        result.confidence = "High";
    }
    else if (result.alignment_inliers >= 8 || result.unchanged_similarity >= 0.76)
    {
        result.confidence = "Medium";
    }
    else
    {
        result.confidence = "Low";
    }

    std::ostringstream summary;
    summary << "Alignment inliers: " << result.alignment_inliers
            << " (" << utils::formatScore(result.alignment_inlier_ratio) << ")"
            << ", changed regions: " << result.changed_regions
            << ", changed area ratio: " << utils::formatScore(result.changed_area_ratio)
            << ", unchanged similarity: " << utils::formatScore(result.unchanged_similarity)
            << ", cleanup consistency: " << utils::formatScore(result.cleanup_consistency);
    result.summary = summary.str();

    return result;
}
