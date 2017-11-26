#pragma once

#include <opencv2/opencv.hpp>
#include "DetectionUnit.h"

cv::Mat getDetectedEdges_8u(cv::Mat &src_8u);

cv::Mat getDistanceTransformFromEdges_32f(cv::Mat &edges_8u);

cv::Mat getDistanceTransform_32f(cv::Mat &src_8u);

int testDetectedEdgesAndDistanceTransform();

float getEdgeOrientation(cv::Mat &srcGray_8u, int x, int y, bool onlyPositive = false);

float getEdgeOrientationFromDistanceTransform(cv::Mat &distTrans_32f, int x, int y, bool onlyPositive = false);

void getEdgeDistAndOri(DetectionUnit &img, int x, int y, float &distance, float &orientation, bool onlyPositive = false);

void showEdgeOrientations(cv::Mat src_8u, cv::Mat edge_8u, bool fromDistanceTransform = false, std::string windowName = "Orientations");