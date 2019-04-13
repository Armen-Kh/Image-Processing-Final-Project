//Crerated by Armen Kh.

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>


//We assume face and body have a similar color ranges for recognizing naked person. Making compliance histogram
std::vector<std::vector<int>> NakedColorsRange(const cv::Mat& img) {
	std::vector<std::vector<int>> ColorHist(3, std::vector<int>(13, 0));

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int p = 0; p < 3; ++p) {
				++ColorHist[p][img.at<cv::Vec3b>(i, j)[p] / 20];
				continue;
			}
		}
	}

	for (int p = 0; p < 3; ++p) {
		std::cout << "\nColorHist [" << p << "] >>>";
		for (int i = 0; i < 13; ++i) {
			std::cout << ColorHist[p][i] << " - ";
		}
	}

	std::vector<std::vector<int>> max3(3, std::vector<int>(3, 0));

	//making max3 array of 3 highest values' indexes
	for (int i = 0; i < 3; ++i) {
		for (int p = 0; p < 3; ++p) {
			max3[i][p] = p;
			for (int j = p + 1; j < 13; ++j) {
				if (ColorHist[i][j] > ColorHist[i][max3[i][p]]) {
					max3[i][p] = j;
				}
			}
			std::swap(ColorHist[i][max3[i][p]], ColorHist[i][p]);
		}
	}

	for (int i = 0; i < 3; ++i) {
		std::cout << "\n";
		for (int j = 0; j < 3; ++j) {
			std::cout << max3[i][j] << " - ";
		}
		std::cout << "\n";
	}

	return max3;
}

//Verification of compliance. Is the man naked?
//The function returns pair, which first value describes state,
//and the second value is the prediction percentage (if state is true).
std::pair<bool, int> NakedChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& face) {

	cv::Mat bodyRoi = img(body);
	std::vector<std::vector<int>> bodyMaxCol = NakedColorsRange(bodyRoi);

	cv::Mat faceRoi = img(face);
	std::vector<std::vector<int>> faceMaxCol = NakedColorsRange(faceRoi);

	std::vector<int> r(3, 0);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int p = 0; p < 3; ++p) {
				if (bodyMaxCol[i][j] == faceMaxCol[i][p]) {
					++r[i];
					std::cout << r[i] << "\n";
				}
			}
		}
	}

	int rSum = r[0] + r[1] + r[2];
	int rNZM = (r[0] != 0) + (r[1] != 0) + (r[2] != 0); // rNZM = r None Zero Members

	std::cout << "\n\n\n\n" << rSum << "=rSum   -   rNZM=" << rNZM << "\n";

	if (rNZM < 3) {
		return std::make_pair(false, 0);
	}
	else {
		return std::make_pair(true, rSum * 10);
	}
}