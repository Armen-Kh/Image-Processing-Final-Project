//Crerated by Armen Kh.

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>

//We assume there are 2 basic white color ranges for recognizing doctors' uniforms. Making compliance histogram
std::vector<int> DoctorPrimaryColorsHistogram(const cv::Mat& img) {
	std::vector<int> ColorHist(3, 0);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<cv::Vec3b>(i, j)[0] > 225 &&
				img.at<cv::Vec3b>(i, j)[1] > 220 &&
				img.at<cv::Vec3b>(i, j)[2] > 220) {
				++ColorHist[0];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 170 && img.at<cv::Vec3b>(i, j)[0] <= 225 &&
				img.at<cv::Vec3b>(i, j)[1] >= 160 && img.at<cv::Vec3b>(i, j)[1] <= 220 &&
				img.at<cv::Vec3b>(i, j)[2] >= 160 && img.at<cv::Vec3b>(i, j)[2] <= 220) {
				++ColorHist[1];
				continue;
			}
			++ColorHist[2];
		}
	}

	return ColorHist;
}


//Verification of compliance. Can a person be a doctor?
//The function returns pair, which first value describes state,
//and the second value is the prediction percentage (if state is true).
std::pair<bool, int> DoctorChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& hat) {

	int rate = 0;

	cv::Mat bodyRoi = img(body);
	std::vector<int> bodyHist = DoctorPrimaryColorsHistogram(bodyRoi);


	cv::Rect bodyMidChestRect(body.x + body.width / 3, body.y, body.width / 3, body.height * 2 / 3);
	cv::Mat bodyMidChestRoi = img(bodyMidChestRect); //Region of interest
	std::vector<int> bodyMidChestHist = DoctorPrimaryColorsHistogram(bodyMidChestRoi);

	for (int i = 0; i < bodyHist.size(); ++i) {
		bodyHist[i] -= bodyMidChestHist[i];
	}

	cv::Mat hatRoi = img(hat);
	std::vector<int> hatHist = DoctorPrimaryColorsHistogram(hatRoi);

	int bodyS = bodyHist[0] + bodyHist[1] + bodyHist[2];
	int bodyColorProp_0 = bodyHist[0] * 100 / bodyS;
	int bodyColorProp_1 = bodyHist[1] * 100 / bodyS;
	int bodyColorProp_2 = bodyHist[2] * 100 / bodyS;
	int bodyColorProp_primary = (bodyS - bodyHist[2]) * 100 / bodyS;

	//statistical datas
	std::cout << "White - " << bodyHist[0] << ",  " << double(bodyHist[0]) / bodyS * 100 << "%\n";
	std::cout << "Dwhite - " << bodyHist[1] << ",  " << double(bodyHist[1]) / bodyS * 100 << "%\n";
	std::cout << "Others - " << bodyHist[2] << ",  " << double(bodyHist[2]) / bodyS * 100 << "%\n";

	int hatS = hatHist[0] + hatHist[1] + hatHist[2];
	int hatColorProp_0 = hatHist[0] * 100 / hatS;
	int hatColorProp_1 = hatHist[1] * 100 / hatS;
	int hatColorProp_2 = hatHist[2] * 100 / hatS;
	int hatColorProp_primary = (hatS - hatHist[2]) * 100 / hatS;

	std::cout << "Doctor hatColor prop = " << hatColorProp_0
		<< " - " << hatColorProp_1
		<< " - " << 100 - hatColorProp_primary << "\n";

	// feature_55 is true if the primary colors is distributed over 55% or more and main color is more than 45%
	bool bodyFeature_55 = (bodyColorProp_primary >= 55 && bodyColorProp_0 >= 45);
	std::cout << "\n=====Doctor bodyFeature_55 = " << bodyFeature_55 << "\n";

	// feature_75 is true if the primary colors is distributed over 75% or more and main color is more than 60%
	bool bodyFeature_75 = (bodyColorProp_primary >= 75 && bodyColorProp_0 >= 60);
	std::cout << "\n=====Doctor bodyFeature_75 = " << bodyFeature_75 << "\n";

	// feature_90 is true if the primary colors is distributed over 90% or more and main color is more than 75%
	bool bodyFeature_90 = (bodyColorProp_primary >= 90 && bodyColorProp_0 >= 75);
	std::cout << "\n=====Doctor bodyFeature_90 = " << bodyFeature_90 << "\n";

	//main compliance checking
	if (bodyColorProp_0 < 45) {
		return std::make_pair(false, 0);
	}

	if (bodyFeature_90) {
		return std::make_pair(true, 80);
	}
	else if (bodyFeature_75) {
		return std::make_pair(true, 65);
	}
	else if (bodyFeature_55) {
		return std::make_pair(true, 50);
	}
	else {
		return std::make_pair(false, 0);
	}
}


