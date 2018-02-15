#pragma once

#include <opencv2/opencv.hpp>
#ifndef _USE_MATH_DEFINES
	#define _USE_MATH_DEFINES
#endif // !_USE_MATH_DEFINES
#include <math.h>
#include <unordered_map>

#include "DetectionUnit.h"
#include "QuantizedTripletValues.h"
#include "TemplateIndex.h"

// CONSTANTS
const cv::Size edgeDetector_BlurSize(3, 3);
const int canny_LowThreshold = 45, canny_Ratio = 5, canny_KernelSize = 3;
const int distanceTransform_MaskSize = 3;

const float thetaD = 3.1;
const float thetaPhi = M_PI / 9.0f;
const float lambda = 0.5f;
const float tau = 0.6;
const float removePixelRatio = 0.5f;
const int kTpl = 4;
const int thetaV = 3;

// Triplets
const int tripletsAmount = 50;
const int pointsInRowCol = 6;
const int pointsEdgeOffset = 4;
const int pointsDistance = 8;

// Sliding window and scale pyramid
const int slidingWindowSize = 48; //Same size as template
const int slidingWindowStep = 3;
const int scalePyramidStep = 1.2;

// Set max to 8, otherwise change hashing for QuantizedTripletValues
const int distanceBins = 4;
const int orientationBins = 6;

const cv::Mat xDerivMask = (cv::Mat_<float>(3, 3) << -1.0f, 0, 1.0f, -2.0f, 0, 2.0f, -1.0f, 0, 1.0f);
const cv::Mat yDerivMask = (cv::Mat_<float>(3, 3) << -1.0f, -2.0f, -1.0f, 0, 0, 0, 1.0f, 2.0f, 1.0f);

typedef std::vector<DetectionUnit> TemplateList;
typedef std::vector<TemplateList> FolderTemplateList;
typedef std::unordered_map<QuantizedTripletValues, std::vector<TemplateIndex>> TemplateHashTable;