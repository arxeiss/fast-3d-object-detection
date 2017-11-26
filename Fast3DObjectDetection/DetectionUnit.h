#pragma once

#include <opencv2/opencv.hpp>

class DetectionUnit
{
public:
	cv::Mat img_8u;
	cv::Mat edges_8u;
	cv::Mat distanceTransform_32f;
	int edgesCount;

	DetectionUnit() {};
	DetectionUnit(cv::Mat img_8u, cv::Mat edges_8u, cv::Mat distanceTransform_32f, int edgesCount) {
		this->img_8u = img_8u;
		this->edges_8u = edges_8u;
		this->distanceTransform_32f = distanceTransform_32f;
		this->edgesCount = edgesCount;
	}
	~DetectionUnit() {
		this->img_8u.release();
		this->edges_8u.release();
		this->distanceTransform_32f.release();
	}
};

