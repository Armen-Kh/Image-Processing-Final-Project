//Crerated by Armen Kh.

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>

//We assume there are 4 basic color ranges for recognizing soldiers' uniforms. Making compliance histogram
std::vector<int> SoldierPrimaryColorsHistogram(const cv::Mat& img) {
	std::vector<int> ColorHist(5, 0);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<cv::Vec3b>(i, j)[0] >= 50 && img.at<cv::Vec3b>(i, j)[0] <= 90 &&
				img.at<cv::Vec3b>(i, j)[1] >= 85 && img.at<cv::Vec3b>(i, j)[1] <= 120 &&
				img.at<cv::Vec3b>(i, j)[2] >= 50 && img.at<cv::Vec3b>(i, j)[2] <= 105) {
				++ColorHist[0];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 100 && img.at<cv::Vec3b>(i, j)[0] <= 155 &&
				img.at<cv::Vec3b>(i, j)[1] >= 140 && img.at<cv::Vec3b>(i, j)[1] <= 200 &&
				img.at<cv::Vec3b>(i, j)[2] >= 150 && img.at<cv::Vec3b>(i, j)[2] <= 210) {
				++ColorHist[1];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 30 && img.at<cv::Vec3b>(i, j)[0] <= 85 &&
				img.at<cv::Vec3b>(i, j)[1] >= 55 && img.at<cv::Vec3b>(i, j)[1] <= 110 &&
				img.at<cv::Vec3b>(i, j)[2] >= 60 && img.at<cv::Vec3b>(i, j)[2] <= 130) {
				++ColorHist[2];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 10 && img.at<cv::Vec3b>(i, j)[0] <= 65 &&
				img.at<cv::Vec3b>(i, j)[1] >= 5 && img.at<cv::Vec3b>(i, j)[1] <= 65 &&
				img.at<cv::Vec3b>(i, j)[2] >= 10 && img.at<cv::Vec3b>(i, j)[2] <= 70) {
				++ColorHist[3];
				continue;
			}
			++ColorHist[4];
		}
	}

	return ColorHist;
}

//Verification of compliance. Can a person be a soldier?
//The function returns pair, which first value describes state,
//and the second value is the prediction percentage (if state is true).
std::pair<bool, int> SoldierChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& hat) {

	int rate = 0;

	cv::Mat bodyRoi = img(body);
	std::vector<int> bodyHist = SoldierPrimaryColorsHistogram(bodyRoi);

	cv::Rect bodyNeckRect(body.x + body.width / 3, body.y, body.width / 3, body.height / 3);
	cv::Mat bodyNeckRoi = img(bodyNeckRect);
	std::vector<int> bodyNeckHist = SoldierPrimaryColorsHistogram(bodyNeckRoi);

	for (int i = 0; i < bodyHist.size(); ++i) {
		bodyHist[i] -= bodyNeckHist[i];
	}

	cv::Mat hatRoi = img(hat);
	std::vector<int> hatHist = SoldierPrimaryColorsHistogram(hatRoi);

	int bodyS = bodyHist[0] + bodyHist[1] + bodyHist[2] + bodyHist[3] + bodyHist[4];
	int bodyColorProp_0 = bodyHist[0] * 100 / bodyS;
	int bodyColorProp_1 = bodyHist[1] * 100 / bodyS;
	int bodyColorProp_2 = bodyHist[2] * 100 / bodyS;
	int bodyColorProp_3 = bodyHist[3] * 100 / bodyS;
	int bodyColorProp_primary = (bodyS - bodyHist[4]) * 100 / bodyS;

	//statistical datas
	std::cout << "Green - " << bodyHist[0] << ",  " << double(bodyHist[0]) / bodyS * 100 << "%\n";
	std::cout << "Cream - " << bodyHist[1] << ",  " << double(bodyHist[1]) / bodyS * 100 << "%\n";
	std::cout << "Brown - " << bodyHist[2] << ",  " << double(bodyHist[2]) / bodyS * 100 << "%\n";
	std::cout << "Dark - " << bodyHist[3] << ",  " << double(bodyHist[3]) / bodyS * 100 << "%\n";
	std::cout << "Others - " << bodyHist[4] << ",  " << double(bodyHist[4]) / bodyS * 100 << "%\n";

	int hatS = hatHist[0] + hatHist[1] + hatHist[2] + hatHist[3] + hatHist[4];
	int hatColorProp_0 = hatHist[0] * 100 / hatS;
	int hatColorProp_1 = hatHist[1] * 100 / hatS;
	int hatColorProp_2 = hatHist[2] * 100 / hatS;
	int hatColorProp_3 = hatHist[3] * 100 / hatS;
	int hatColorProp_primary = (hatS - hatHist[4]) * 100 / hatS;

	std::cout << "hatColor prop = " << hatColorProp_0
		<< " - " << hatColorProp_1
		<< " - " << hatColorProp_2
		<< " - " << hatColorProp_3
		<< " - " << 100 - hatColorProp_primary << "\n";

	// feature_70 is true if none of the primary colors is distributed over 70% or more
	bool bodyFeature_70 = 1;
	if (bodyColorProp_primary != 0) {
		bodyFeature_70 = ((bodyColorProp_0 * 100 / bodyColorProp_primary < 70) &&
			(bodyColorProp_1 * 100 / bodyColorProp_primary < 70) &&
			(bodyColorProp_2 * 100 / bodyColorProp_primary < 70) &&
			(bodyColorProp_3 * 100 / bodyColorProp_primary < 70));
	}
	std::cout << "\n=====bodyFeature_70 = " << bodyFeature_70 << "\n";

	//feature_adc is true if at least two primary colors have a more or equal average distribution
	int bodyAverDist = bodyColorProp_primary / 4;
	int bodyFeature_adc = (bodyColorProp_0 >= bodyAverDist) +
		(bodyColorProp_1 >= bodyAverDist) +
		(bodyColorProp_2 >= bodyAverDist) +
		(bodyColorProp_3 >= bodyAverDist);
	std::cout << "\n=====bodyFeature_adc = " << bodyFeature_adc << "\n";

	//feature_hat is true if hypothetical hat is exist and has 35% or more primary colors proportion
	bool hatFeature_35 = hatColorProp_primary >= 35;
	std::cout << "\n=====hatFeature_35 = " << hatFeature_35 << "\n";

	//main compliance checking
	if (bodyColorProp_primary <= 40) {
		return std::make_pair(false, 0);
	}

	if (!bodyFeature_70) {
		if (hatFeature_35) {
			return std::make_pair(true, 0); //return std::make_pair(false, 0); 
		}
		else {
			return std::make_pair(false, 0);
		}
	}

	int h = 0;
	if (hatFeature_35) h = 5;

	if (bodyColorProp_primary < 50) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 40 + h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 35 + h);
		else return std::make_pair(true, 15 + h);
	}
	else if (bodyColorProp_primary < 60) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 50 + 2 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 45 + 2 * h);
		else return std::make_pair(true, 25 + 2 * h);
	}
	else if (bodyColorProp_primary < 70) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 65 + 3 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 60 + 3 * h);
		else return std::make_pair(true, 40 + 3 * h);
	}
	else if (bodyColorProp_primary < 80) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 70 + 3 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 65 + 3 * h);
		else return std::make_pair(true, 45 + 3 * h);
	}
	else {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 80 + 4 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 75 + 4 * h);
		else return std::make_pair(true, 60 + 4 * h);
	}
}
