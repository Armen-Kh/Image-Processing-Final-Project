//IP_Project Specialty Recognition
//Crerated by Armen and Amalia (13th week)

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>

#include "functions.h"

//This function for futures testing
std::vector<int> ColorStatistic(const cv::Mat& img) {
	long q = img.rows * img.cols;
	std::vector<int> cv(9, 0);
	std::vector<long long> av(3, 0);
	std::vector<int> min(3, 0);
	std::vector<int> max(3, 0);

	for (int p = 0; p < 3; ++p) {
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				av[p] += img.at<cv::Vec3b>(i, j)[p];
				if (img.at<cv::Vec3b>(i, j)[p] < min[p]) min[p] = img.at<cv::Vec3b>(i, j)[0];
				if (img.at<cv::Vec3b>(i, j)[p] > max[p]) max[p] = img.at<cv::Vec3b>(i, j)[0];
			}
		}
		cv[p] = int(av[p] / q);
	}

	cv[3] = min[0];
	cv[4] = min[1];
	cv[5] = min[2];
	cv[6] = max[0];
	cv[7] = max[1];
	cv[8] = max[2];

	std::cout << "\n============COLOR STATISTIC==============="
		<< "\n average B = " << cv[0]
		<< "\n average G = " << cv[1]
		<< "\n average R = " << cv[2]
		<< "\n min B = " << cv[3]
		<< "\n min G = " << cv[4]
		<< "\n min R = " << cv[5]
		<< "\n max B = " << cv[6]
		<< "\n max G = " << cv[7]
		<< "\n max R = " << cv[8]
		<< "\n============COLOR STATISTIC===============\n";

	return cv;
}

cv::Mat PredictionImage(char c, bool b, int i) {
	srand((unsigned int)time(0));
	std::vector<cv::Mat> rate;
	for (int i = 0; i < 21; ++i) {
		std::string name = cv::format("%d.jpg", i);
		cv::Mat img = cv::imread(name);
		if (img.empty())
		{
			std::cerr << "whaa " << name << " can't be loaded!" << std::endl;
			continue;
		}
		rate.push_back(img);
	}

	std::vector<cv::Mat> text;
	for (int i = 0; i < 6; ++i) {
		std::string name = cv::format("%d.jpg", i + 21);
		cv::Mat img = cv::imread(name);
		if (img.empty())
		{
			std::cerr << "whaa " << name << " can't be loaded!" << std::endl;
			continue;
		}
		text.push_back(img);
	}

	if (!b) {
		return text[3 + rand() % 3];
	}

	cv::Mat result;
	switch (c)
	{
	case 'n': {vconcat(text[0], rate[i / 5], result); return result; }
	case 'd': {vconcat(text[1], rate[i / 5], result); return result; }
	case 's': {vconcat(text[2], rate[i / 5], result); return result; }
	}
}

int main() {
	cv::Mat img = cv::imread("therok.webp");

	cv::namedWindow("original_img", cv::WINDOW_AUTOSIZE);
	cv::imshow("original_img", img);

	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(grayImg, grayImg);
	cv::imwrite("...mgray.jpg", grayImg);

	cv::CascadeClassifier face_cascade;
	std::vector<cv::Rect> face;

	if (!face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml")) {
		std::cout << "===LOADING ERROR(*_1.xml)===\n";
		return -1;
	};

	//=====face detection 1=====
	face_cascade.detectMultiScale(grayImg, face, 1.1, 3, 0, cv::Size(30, 30));

	if (!face.size()) {
		if (!face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml")) {
			std::cout << "===LOADING ERROR(*_2.xml)===\n";
			return -1;
		};
		//=====face detection 2=====
		face_cascade.detectMultiScale(grayImg, face, 1.1, 3, 0, cv::Size(30, 30));

		if (!face.size()) {
			std::cout << "=====SORRY, THE OBJECT WAS NOT FOUND=====\n";
			cv::Mat info = cv::imread("nwf.jpg");
			cv::namedWindow("info", cv::WINDOW_AUTOSIZE);
			cv::imshow("info", info);
			cv::waitKey(0);
			return 0;
		}
	}

	cv::Rect faceRect = face[0];
	cv::Mat faceRoi = img(faceRect);

	std::cout << "\n face color statistic \n";
	//ColorStatistic(faceRoi);
	//NakedColorsRange(faceRoi);

	//=====face marking=====
	cv::Mat markedImg = img.clone();
	cv::rectangle(markedImg, faceRect, cv::Scalar(0, 255, 0), 2);
	//cv::namedWindow("marked_img", cv::WINDOW_AUTOSIZE);
	//cv::imwrite("marked.jpg", markedImg);
	//cv::imshow("marked_img", markedImg);

	//=====upperbody detection=====
	int upperBodyUpperLeft_x = std::max(0, int(faceRect.x - faceRect.width / 2));
	int upperBodyUpperLeft_y = std::min(img.rows - 1, int(faceRect.y + faceRect.height * 1.3));
	int upperBody_width = std::min(img.cols - upperBodyUpperLeft_x, faceRect.width * 2);
	int upperBody_height = std::min(img.rows - upperBodyUpperLeft_y, int(faceRect.height * 2));

	cv::Rect upperBodyRect(upperBodyUpperLeft_x, upperBodyUpperLeft_y, upperBody_width, upperBody_height);
	cv::Mat upperBodyRoi = img(upperBodyRect);

	std::cout << "\n body color statistic \n";
	//ColorStatistic(upperBodyRoi);
	//NakedColorsRange(upperBodyRoi);

	//=====upperbody marking=====
	cv::rectangle(markedImg, upperBodyRect, cv::Scalar(0, 0, 255), 2);

	//=====hat detection=====
	int hatUpperLeft_x = faceRect.x + int(faceRect.width * 0.05);
	int hatUpperLeft_y = std::max(0, int(faceRect.y - (double)faceRect.height * 0.15));
	int hat_width = int(faceRect.width * 0.9);
	int hat_height = int(faceRect.y - hatUpperLeft_y + (double)faceRect.height * 0.1);

	cv::Rect hatRect(hatUpperLeft_x, hatUpperLeft_y, hat_width, hat_height);
	cv::Mat hatRoi = img(hatRect);

	//=====hat marking=====
	cv::rectangle(markedImg, hatRect, cv::Scalar(255, 0, 0), 2);

	cv::namedWindow("marked_img", cv::WINDOW_NORMAL);
	cv::imwrite("...marked.jpg", markedImg);
	cv::imshow("marked_img", markedImg);

	std::pair<bool, int> SoldierResult = SoldierChecking(img, upperBodyRect, hatRect);
	std::cout << "_______________________SoldierResult=(" << SoldierResult.first << ", " << SoldierResult.second << "%)\n";

	std::pair<bool, int> DoctorResult = DoctorChecking(img, upperBodyRect, hatRect);
	std::cout << "_______________________DoctorResult=(" << DoctorResult.first << ", " << DoctorResult.second << "%)\n";

	std::pair<bool, int> NakedResult = NakedChecking(img, upperBodyRect, faceRect);
	std::cout << "_______________________NakedResult=(" << NakedResult.first << ", " << NakedResult.second << "%)\n";

	cv::Mat resultImg;

	if (NakedResult.second >= DoctorResult.second && NakedResult.second >= SoldierResult.second) {
		resultImg = PredictionImage('n', NakedResult.first, NakedResult.second);
	}
	else if (DoctorResult.second >= SoldierResult.second) {
		resultImg = PredictionImage('d', DoctorResult.first, DoctorResult.second);
	}
	else {
		resultImg = PredictionImage('s', SoldierResult.first, SoldierResult.second);
	}

	cv::namedWindow("marked_img", cv::WINDOW_AUTOSIZE);
	cv::imwrite("result.jpg", resultImg);
	cv::imshow("marked_img", resultImg);
	cv::waitKey(0);

	return 0;
}
