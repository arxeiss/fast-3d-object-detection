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

// 0 - auto by quantile
// 1 - hardcoded
// 2 - all bins same range
#define HASH_SETTINGS_DIST_BINS_METHOD 0

// CONSTANTS
const cv::Size edgeDetector_BlurSize(5, 5);
const int canny_LowThreshold = 45, canny_Ratio = 3, canny_KernelSize = 3;
const int distanceTransform_MaskSize = 3;

const float thetaD = 3.1;			// Used in removing edges and computing chamfer score
const float thetaPhi = M_PI / 9.0f; // Same as above
const float lambda = 0.5f;			// Used in chamfer score computing - Compensating the Bias Towards Simples Shapes
const float tau = 0.6;				// Ratio * kTpl - how many edges must be in kTpl templates same to keep them
const float removePixelRatio = 0.4f;// p (paper) - How many percent of edges to remove
const int kTpl = 4;					// k (paper) - How many templates are compared to current to remove non stable edges
const int thetaV = 3;				// How many triplets must have same template index to keep it as candidate
const float minEdgesRatio = 0.95;	// Min edges to keep candidate (minEdgesRatio * minEdges)
const int minBGColorThreshold = 37; // t_b (paper) -  Minimal uchar value on background in loaded templates 
const int minQuadrantEdges = 10;	// e_r
const float minChamferScore = 0.5; // t_c

// Triplets
const int tripletsAmount = 50; // L
const int pointsInRowCol = 6; // M = 36 (6*6)
const int pointsEdgeOffset = 4;
const int pointsDistance = 8;

// Sliding window, scale pyramid and Ground Truth
const int slidingWindowSize = 48; //Same size as template
const int slidingWindowStep = 3; // s_s
const int scalePyramidSteps = 10; // s_n
const float scalePyramidResizeRatio = 1.2; // s_r
const float NMSMinOverlap = 0.5f; // o_m
const float GTMinOverlap = 0.5f; // o_g

// Set max to 8, otherwise change hashing for QuantizedTripletValues
const int distanceBins = 4; // n_d
const int orientationBins = 6; // n_phi

const cv::Mat xDerivMask = (cv::Mat_<float>(3, 3) << -1.0f, 0, 1.0f, -2.0f, 0, 2.0f, -1.0f, 0, 1.0f);
const cv::Mat yDerivMask = (cv::Mat_<float>(3, 3) << -1.0f, -2.0f, -1.0f, 0, 0, 0, 1.0f, 2.0f, 1.0f);

typedef std::vector<DetectionUnit> TemplateList;
typedef std::vector<TemplateList> FolderTemplateList;
typedef std::unordered_map<QuantizedTripletValues, std::vector<TemplateIndex>> TemplateHashTable;