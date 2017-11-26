#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<float> getImgVector(cv::Mat img_32f, int step, int margin, int startX = 0, int startY = 0, int width = -1, int height = -1);
int compareVectors(std::vector<float> v1, std::vector<float> v2, float thetaD, float thetaPhi);
