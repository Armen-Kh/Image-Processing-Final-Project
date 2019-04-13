//#pragma once

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

//We assume there are 4 basic color ranges for recognizing soldiers' uniforms. Making compliance histogram
std::vector<int> SoldierPrimaryColorsHistogram(const cv::Mat& img);

//Verification of compliance. Can a person be a soldier?
//The function returns pair, which first value describes state,
//and the second value is the prediction percentage (if state is true).
std::pair<bool, int> SoldierChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& hat);

//We assume there are 2 basic white color ranges for recognizing doctors' uniforms. Making compliance histogram
std::vector<int> DoctorPrimaryColorsHistogram(const cv::Mat& img);

//Verification of compliance. Can a person be a doctor?
//The function returns pair, which first value describes state,
//and the second value is the prediction percentage (if state is true).
std::pair<bool, int> DoctorChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& hat);

//We assume face and body have a similar color ranges for recognizing naked person. Making compliance histogram
std::vector<std::vector<int>> NakedColorsRange(const cv::Mat& img);

//Verification of compliance. Is the man naked?
//The function returns pair, which first value describes state,
//and the second value is the prediction percentage (if state is true).
std::pair<bool, int> NakedChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& face);

#endif
