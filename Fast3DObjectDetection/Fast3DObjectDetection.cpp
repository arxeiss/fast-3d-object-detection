
#include <stdio.h>
#include <tchar.h>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <iomanip>
#include <unordered_map>

#include <omp.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "utils.h"
#include "visualize.h"

#include "DetectionUnit.h"
#include "TemplateChamferScore.h"
#include "Triplet.h"
#include "TemplateIndex.h"
#include "TripletValues.h"
#include "QuantizedTripletValues.h"
#include "HashSettings.h"
#include "TimeMeasuring.h"

typedef std::vector<DetectionUnit> TemplateList;
typedef std::vector<TemplateList> FolderTemplateList;
typedef std::unordered_map<QuantizedTripletValues, std::vector<TemplateIndex>> TemplateHashTable;

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

const int tripletsAmount = 50;
const int pointsInRowCol = 6;
const int pointsEdgeOffset = 4;
const int pointsDistance = 8;

// Set max to 8, otherwise change hashing for QuantizedTripletValues
const int distanceBins = 4;
const int orientationBins = 6;

const cv::Mat xDerivMask = (cv::Mat_<float>(3, 3) << -1.0f, 0, 1.0f, -2.0f, 0, 2.0f, -1.0f, 0, 1.0f);
const cv::Mat yDerivMask = (cv::Mat_<float>(3, 3) << -1.0f, -2.0f, -1.0f, 0, 0, 0, 1.0f, 2.0f, 1.0f);

// --------- edges + distance transform -----------
cv::Mat getDetectedEdges_8u(cv::Mat &src_8u) {
	cv::Mat detected_edges_8u, dst;

	/// Reduce noise with a kernel 3x3
	blur(src_8u, detected_edges_8u, edgeDetector_BlurSize);
	/// Canny detector - need 8U matrix
	cv::Canny(detected_edges_8u, detected_edges_8u, canny_LowThreshold, canny_LowThreshold*canny_Ratio, canny_KernelSize);

	/// Using Canny's output as a mask, we display our result
	// Clear img
	dst = cv::Mat(src_8u.size(), CV_8UC1);
	dst = cv::Scalar(255);
	cv::Mat blackImage = cv::Mat::zeros(src_8u.size(), CV_8UC1);
	blackImage.copyTo(dst, detected_edges_8u);

	return dst;
}

cv::Mat getDistanceTransformFromEdges_32f(cv::Mat &edges_8u) {
	cv::Mat distTransform_32f;
	cv::distanceTransform(edges_8u, distTransform_32f, CV_DIST_L2, distanceTransform_MaskSize); // return 32F matrix
	return distTransform_32f;
}

cv::Mat getDistanceTransform_32f(cv::Mat &src_8u) {
	
	cv::Mat edges_8u = getDetectedEdges_8u(src_8u);
	cv::Mat distTransform_32f = getDistanceTransformFromEdges_32f(edges_8u);

	// Normalize the distance image for range = {0.0, 1.0} only for viewing.. otherwise it destroys everything
	/*normalize(dst, dst, 0.0f, 1.0f, cv::NORM_MINMAX);
	imshow("dst", dst);*/
	return distTransform_32f;
}
/// TEST func - potrebuje ty 2 vyse
int testDetectedEdgesAndDistanceTransform() {
	cv::Mat rect_8u = cv::imread("images/rect.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat dstTransform = getDistanceTransform_32f(rect_8u), dstTransformNorm;

	cv::normalize(dstTransform, dstTransformNorm, 0.0f, 1.0f, cv::NORM_MINMAX);
	
	for (int x = 5; x < dstTransform.cols; x += 30)
	{
		for (int y = 5; y < dstTransform.rows; y += 20)
		{
			std::stringstream ss;
			ss << std::fixed << std::setprecision(1) << dstTransform.at<float>(y, x);
			cv::circle(dstTransform, cv::Point(x, y), 1, cv::Scalar(0.5));
			cv::circle(dstTransformNorm, cv::Point(x, y), 1, cv::Scalar(0.5));
			cv::putText(dstTransform, ss.str(), cv::Point(x + 2, y), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0.5));
		}
	}
	int x = 11, y = 15;
	std::stringstream ss;
	ss << std::fixed << std::setprecision(10) << dstTransform.at<float>(y, x);
	cv::circle(dstTransform, cv::Point(x, y), 3, cv::Scalar(0.5));
	cv::circle(dstTransformNorm, cv::Point(x, y), 3, cv::Scalar(0.5));
	cv::putText(dstTransform, ss.str(), cv::Point(x + 2, y), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0.5));

	cv::imshow("distTransform", dstTransform);
	cv::imshow("distTransformNorm", dstTransformNorm);
	cv::waitKey(0);
	return 1;
}

// --------- edge distance and orientation -----------
float getEdgeOrientation(cv::Mat &srcGray_8u, int x, int y, bool onlyPositive = false) {
	if (x + 1 >= srcGray_8u.cols || x < 1 || y < 1 || y + 1 >= srcGray_8u.rows) {
		return 0.0f;
	}
	float sXVal = 0.0f;
	float sYVal = 0.0f;
	for (int mX = 0; mX < 3; mX++)
	{
		for (int mY = 0; mY < 3; mY++)
		{
			//float cVal = getPixelAsfloat(srcGray_8u, x + mX - 1, y + mY - 1);
			float cVal = ((float)srcGray_8u.at<uchar>(y + mY - 1, x + mX - 1)) / 255.0f;
			sXVal += cVal * xDerivMask.at<float>(mY, mX);
			sYVal += cVal * yDerivMask.at<float>(mY, mX);
		}
	}
	// With multiplying, we get same result..
	/*sXVal *= 1.0f / 8.0f;
	sYVal *= 1.0f / 8.0f;*/
	float angle = atan2(sYVal, sXVal);
	if (angle < 0 && onlyPositive) {
		angle += M_PI;
	}
	return angle;
}

float getEdgeOrientationFromDistanceTransform(cv::Mat &distTrans_32f, int x, int y, bool onlyPositive = false) {
	float pointDistance = distTrans_32f.at<float>(y, x);
	float angle = atan2(pointDistance - (float)distTrans_32f.at<float>(y - 1, x), pointDistance - (float)distTrans_32f.at<float>(y, x - 1));
	if (angle < 0 && onlyPositive)
	{
		angle += M_PI;
	}
	return angle;
}

void getEdgeDistAndOri(DetectionUnit &img, int x, int y, float &distance, float &orientation, bool onlyPositive = false) {
	distance = img.distanceTransform_32f.at<float>(y, x);
	orientation = (distance == 0.0f) ?
		getEdgeOrientation(img.img_8u, x, y, onlyPositive) :
		getEdgeOrientationFromDistanceTransform(img.distanceTransform_32f, x, y, onlyPositive);
}

/// TEST func -- potrebuje edges+distance transform; edge distance and orientation; 
void showEdgeOrientations(cv::Mat src_8u, cv::Mat edge_8u, bool fromDistanceTransform = false, std::string windowName = "Orientations") {
	cv::Mat edgeWithOri, distanceTransform_32f;
	cv::cvtColor(edge_8u, edgeWithOri, CV_GRAY2BGR);

	if (fromDistanceTransform) {
		distanceTransform_32f = getDistanceTransformFromEdges_32f(edge_8u);
	}

	bool onlyPositive = false;

	for (int y = 5; y < edge_8u.rows; y += 5)
	{
		for (int x = 0; x < edge_8u.cols; x++)
		{
			if (edge_8u.at<uchar>(y, x) > 0)
			{
				continue;
			}
			//cv::circle(edgeWithOri, cv::Point(x, y), 1, cv::Scalar(1));
			if (fromDistanceTransform)
			{
				int offsetX = x - 6, offsetY = y - 6;
				float angle = getEdgeOrientationFromDistanceTransform(distanceTransform_32f, offsetX, offsetY, onlyPositive);
				int dX = offsetX - round(cos(angle) * 6);
				int dY = offsetY - round(sin(angle) * 6);
				cv::line(edgeWithOri, cv::Point(offsetX, offsetY), cv::Point(dX, dY), cv::Scalar(0, 0, 255));
				cv::circle(edgeWithOri, cv::Point(offsetX, offsetY), 1, cv::Scalar(255, 0, 0), -1);
			}
			else {
				float angle = getEdgeOrientation(src_8u, x, y, onlyPositive);
				int dX = x - round(cos(angle) * 6);
				int dY = y - round(sin(angle) * 6);
				cv::line(edgeWithOri, cv::Point(x, y), cv::Point(dX, dY), cv::Scalar(0, 0, 255));
			}
			x += 4;
		}
	}

	cv::imshow(windowName, edgeWithOri);
}

//  --------------- Preprocessing -------------------
// Remove 2 highest bins
cv::Mat removeDiscriminatePoints_8u(cv::Mat &src_8u, cv::Mat &edge_8u, float removePixelRatio) {
	cv::Mat filtered_8u;
	edge_8u.copyTo(filtered_8u);
	// 15, 45, 75, 105, 135, 165
	// 0-30, 30-60, 60-90, 90-120, 120-150, 150
	int const histogramBins = 6;
	int const removeBins = 2;
	float const ranges[histogramBins - 1] = { M_PI/6.0f, M_PI/3.0f, M_PI/2.0f, 2*M_PI/3.0f, 5*M_PI/6.0f };

	std::vector<cv::Point2i> histogram[histogramBins];
	int pixelsToCnt = 0;

	int offset = 1;	
	for (int x = offset; x < edge_8u.cols - offset; x++)
	{
		for (int y = offset; y < edge_8u.rows - offset; y++)
		{
			uchar pixelVal = edge_8u.at<uchar>(y, x);
			if (pixelVal >= 1)
			{
				continue;
			}
			
			pixelsToCnt++;
			int histBinIndex = histogramBins - 1;
			float angle = getEdgeOrientation(src_8u, x, y, true);
			for (int i = 0; i < histogramBins - 1; i++)
			{
				if (angle < ranges[i])
				{
					histBinIndex = i;
					break;
				}
			}
			histogram[histBinIndex].push_back(cv::Point2i(x, y));
		}
	}

	/*for (int i = 0; i < histogramBins; i++)
	{
		std::printf("%d: %d\n", i, histogram[i].size());
	}*/

	for (int rm = 0; rm < removeBins; rm++) {
		int maxBinI = 0, maxBinCnt = 0;
		for (int i = 0; i < histogramBins; i++)
		{
			if (maxBinCnt < histogram[i].size())
			{
				maxBinCnt = histogram[i].size();
				maxBinI = i;
			}
		}

		//printf("Remove %d. bin\n", maxBinI);

		int pixelsToRemove = ((float)histogram[maxBinI].size()) * removePixelRatio;
		std::random_shuffle(histogram[maxBinI].begin(), histogram[maxBinI].end());

		for (int i = 0; i < pixelsToRemove; i++)
		{
			cv::Point2i point = histogram[maxBinI].at(i);
			filtered_8u.at<uchar>(point) = 255;
		}
		histogram[maxBinI].clear();
	}

	return filtered_8u;

}
/// TEST func - edges + distance transform;
int selectingByEdgeOrientations() {
	cv::Mat hand_8u = cv::imread("images/hand.png", CV_LOAD_IMAGE_GRAYSCALE);
	printMatType(hand_8u);
	
	cv::Mat handEdge_8u = getDetectedEdges_8u(hand_8u);
	cv::Mat handEdgeFiltered = removeDiscriminatePoints_8u(hand_8u, handEdge_8u, removePixelRatio);
	
	//cv::imshow("hand", hand);
	cv::imshow("hand_edge", handEdge_8u);
	cv::imshow("hand_edge_filtered", handEdgeFiltered);

	cv::waitKey(0);
	return 0;
}

// detect edges, distance transform, count edges
void prepareDetectionUnit(DetectionUnit &dt, bool renewEdges = false, bool renewDistTransform = false, bool recountEdges = false) {
	if (dt.edges_8u.empty() || renewEdges)
	{
		dt.edges_8u = getDetectedEdges_8u(dt.img_8u);
		recountEdges = true;
	}
	if (dt.distanceTransform_32f.empty() || renewDistTransform)
	{
		dt.distanceTransform_32f = getDistanceTransformFromEdges_32f(dt.edges_8u);
	}
	if (dt.edgesCount == 0 || recountEdges) {
		for (int x = 0; x < dt.edges_8u.cols; x++)	
		{
			for (int y = 0; y < dt.edges_8u.rows; y++)
			{
				if (dt.edges_8u.at<uchar>(y, x) == 0) {
					dt.edgesCount++;
				}
			}
		}
	}
}

//  --------------- Compare -------------------
float getOrientedChamferScore(DetectionUnit &srcTemplate, DetectionUnit &comparingImage, float averageEdges, float lambda, float thetaD, float thetaPhi, int* matchEdges = NULL) {
	int edges = 0;
	for (int x = 0; x < srcTemplate.edges_8u.cols; x++)
	{
		for (int y = 0; y < srcTemplate.edges_8u.rows; y++)
		{
			if (srcTemplate.edges_8u.at<uchar>(y, x) > 0)
			{
				continue;
			}
			float distance = comparingImage.distanceTransform_32f.at<float>(y, x);
			if (distance > thetaD) {
				continue;
			}
			float angleT = getEdgeOrientation(srcTemplate.img_8u, x, y, true);
			float angleI = (distance == 0.0f) ?
				getEdgeOrientation(comparingImage.img_8u, x, y, true) :
				getEdgeOrientationFromDistanceTransform(srcTemplate.distanceTransform_32f, x, y, true);
			if (abs(angleT - angleI) <= thetaPhi)
			{
				edges++;
			}
		}
	}
	float denominator = (lambda * (float)srcTemplate.edgesCount) + ((1 - lambda) * averageEdges);
	if (matchEdges != NULL) {
		*matchEdges = edges;
	}
	return (float)edges / denominator ;
}

int compareDescChamferScore(TemplateChamferScore &a, TemplateChamferScore &b)
{
	// Reverse order
	return !(a.chamferScore < b.chamferScore);
}

//  --------------- Preprocessing -------------------
cv::Mat removeNonStablePoints_8u(DetectionUnit &srcTemplate, std::vector<DetectionUnit> &simmilarTemplates, float thetaD, float thetaPhi, float tau) {
	int kTpl = simmilarTemplates.size();
	cv::Mat removedPoints_8u;
	srcTemplate.edges_8u.copyTo(removedPoints_8u);
	for (int x = 0; x < srcTemplate.edges_8u.cols; x++)
	{
		for (int y = 0; y < srcTemplate.edges_8u.rows; y++)
		{
			if (srcTemplate.edges_8u.at<uchar>(y, x) > 0)
			{
				continue;
			}
			int edges = 0;
			for (int s = 0; s < kTpl; s++)
			{
				float distance = simmilarTemplates[s].distanceTransform_32f.at<float>(y, x);
				if (distance > thetaD) {
					continue;
				}
				float angleT = getEdgeOrientation(simmilarTemplates[s].img_8u, x, y, true);
				float angleI = (distance == 0.0f) ?
					getEdgeOrientation(simmilarTemplates[s].img_8u, x, y, true) :
					getEdgeOrientationFromDistanceTransform(simmilarTemplates[s].distanceTransform_32f, x, y, true);
				if (abs(angleT - angleI) <= thetaPhi)
				{
					edges++;
				}
			}
			if ((float)edges < tau * (float)kTpl )
			{
				removedPoints_8u.at<uchar>(y, x) = 255;
			}
		}
	}
	return removedPoints_8u;
}

void filterTemplateEdges(std::vector<DetectionUnit> &templates, int kTpl, float lambda, float thetaD, float thetaPhi, float tau, float removePixelRatio) {
	int totalEdges = 0;
	for (int i = 0; i < templates.size(); i++)
	{
		totalEdges += templates[i].edgesCount;
	}
	
	float averageEdges = (float)totalEdges / (float)templates.size();
	std::vector<cv::Mat> removedEdges(templates.size());
	#pragma omp parallel for
	for (int t = 0; t < templates.size(); t++)
	{
		std::vector<TemplateChamferScore> simmilarity;
		for (int i = 0; i < templates.size(); i++)
		{
			if (i != t)
			{
				float chamferScore = getOrientedChamferScore(templates[t], templates[i], averageEdges, lambda, thetaD, thetaPhi);
				simmilarity.push_back(TemplateChamferScore(i, chamferScore));
			}
		}
		std::sort(simmilarity.begin(), simmilarity.end(), compareDescChamferScore);

		/*showResized("src",  templates[t].edges_8u, 3);
		for (int i = 0; i < 10; i++)
		{
			showResized(std::to_string(i)+"sim", templates[simmilarity[i].index].edges_8u, 3);
		}
		cv::waitKey();*/


		std::vector<DetectionUnit> simmilarTemplates(kTpl);
		for (int i = 0; i < kTpl; i++)
		{
			simmilarTemplates[i] = templates[simmilarity[i].index];
		}
		cv::Mat nonStableEdges = removeNonStablePoints_8u(templates[t], simmilarTemplates, thetaD, thetaPhi, tau);
		removedEdges[t] = removeDiscriminatePoints_8u(templates[t].img_8u, nonStableEdges, removePixelRatio);
		
		/*showResized("srcEdges", templates[t].edges_8u, 3);
		cv::Mat nonStableEdges = removeNonStablePoints_8u(templates[t], simmilarTemplates, thetaD, thetaPhi, tau);
		showResized("non stable", nonStableEdges, 3);
		removedEdges[t] = removeDiscriminatePoints_8u(templates[t].img_8u, nonStableEdges, removePixelRatio);
		showResized("discrimintaive", removeDiscriminatePoints_8u(templates[t].img_8u, templates[t].edges_8u, removePixelRatio), 3);
		showResized("total", removedEdges[t], 3);
		cv::waitKey(0);*/
		/*if (t % 10 == 0)
		{
			std::printf("Filtered %d / %d templates\r", t, templates.size());
		}*/
	}
	#pragma omp parallel for
	for (int t = 0; t < templates.size(); t++) {
		templates[t].edges_8u = removedEdges[t];
		prepareDetectionUnit(templates[t], false, false, true);
	}
}

std::vector<Triplet> generateTriplets(const int amount, const int inColRow, const int edgeOffset, const int pointsDistance)
{
	const int tripletPoints = inColRow * inColRow;
	std::vector<Triplet> triplets;
	for (int i = 0; i < amount; i++)
	{
		int p1 = rand() % tripletPoints, p2 = rand() % tripletPoints, p3 = rand() % tripletPoints;
		while (p1 == p2)
		{
			p2 = rand() % tripletPoints;
		}
		while (p1 == p3 || p2 == p3)
		{
			p3 = rand() % tripletPoints;
		}
		
		Triplet newTriplet(cv::Point((p1 % inColRow) * pointsDistance + edgeOffset, (p1 / inColRow) * pointsDistance + edgeOffset),
			cv::Point((p2 % inColRow) * pointsDistance + edgeOffset, (p2 / inColRow) * pointsDistance + edgeOffset),
			cv::Point((p3 % inColRow) * pointsDistance + edgeOffset, (p3 / inColRow) * pointsDistance + edgeOffset));
		bool collision = false;
		for (int t = 0; t < triplets.size(); t++)
		{
			if (triplets[t] == newTriplet)
			{
				collision = true;
				break;
			}
		}
		if (collision)
		{
			i--;
			continue;
		}
		triplets.push_back(newTriplet);
	}
	return triplets;
}

int loadAllTemplates(FolderTemplateList &templates) {
	std::string folders[] = {
		"images/CMP-8objs/train-opt2/block/",
		"images/CMP-8objs/train-opt2/bridge/",
		"images/CMP-8objs/train-opt2/cup/",
		"images/CMP-8objs/train-opt2/driver/",
		"images/CMP-8objs/train-opt2/eye/",
		"images/CMP-8objs/train-opt2/lid/",
		"images/CMP-8objs/train-opt2/screw/",
		"images/CMP-8objs/train-opt2/whiteblock/"
	};
	int templatesInFolder = 1620;
	templates = FolderTemplateList(sizeof(folders)/sizeof(folders[0]));
	int templatesLoaded = 0;
	
	#pragma omp parallel for
	for (int f = 0; f < templates.size(); f++)
	{
		templates[f] = TemplateList(templatesInFolder);
		TimeMeasuring elapsedTime(true);
		
		for (int t = 1; t <= templatesInFolder; t++)
		{
			DetectionUnit unit{};
			std::string templateName = std::to_string(t);
			templateName.insert(templateName.begin(), 5 - templateName.size(), '0');
			unit.img_8u = cv::imread(folders[f] + "template_" + templateName + ".png", CV_LOAD_IMAGE_GRAYSCALE);
			prepareDetectionUnit(unit);
			templates[f][t-1] = unit;
			#pragma omp critical
			templatesLoaded++;
		}

		std::printf("Folder \"%s\" loaded in %d [ms]\n", folders[f].c_str(), elapsedTime.getTimeFromBeginning());

	}
	return templatesLoaded;
}

void countTripletsValues(std::vector<TripletValues> &tripletsValues, FolderTemplateList &templates, std::vector<Triplet> &triplets,
	int templatesLoaded, float *minD = NULL, float *maxD = NULL, float *minPhi = NULL, float *maxPhi = NULL)
{
	float minDTmp = FLT_MAX, maxDTmp = FLT_MIN, minPhiTmp = FLT_MAX, maxPhiTmp = FLT_MIN;
	tripletsValues = std::vector<TripletValues>(templatesLoaded * triplets.size());
	std::vector<float> distances = std::vector<float>(tripletsValues.size() * 3);

	int templateTripletValI = 0;
	int distancesI = 0;
	for (int f = 0; f < templates.size(); f++)
	{
		TemplateList *listTmpPtr = &templates[f];
		for (int tpl = 0; tpl < listTmpPtr->size(); tpl++)
		{
			for (int trp = 0; trp < triplets.size(); trp++)
			{
				TripletValues trpVal(trp, TemplateIndex(f, tpl));
				getEdgeDistAndOri((*listTmpPtr)[tpl], triplets[trp].p1.x, triplets[trp].p1.y, trpVal.d1, trpVal.phi1, true);
				getEdgeDistAndOri((*listTmpPtr)[tpl], triplets[trp].p2.x, triplets[trp].p2.y, trpVal.d2, trpVal.phi2, true);
				getEdgeDistAndOri((*listTmpPtr)[tpl], triplets[trp].p3.x, triplets[trp].p3.y, trpVal.d3, trpVal.phi3, true);
				tripletsValues[templateTripletValI] = trpVal;
				templateTripletValI++;

				distances[distancesI++] = trpVal.d1;
				distances[distancesI++] = trpVal.d2;
				distances[distancesI++] = trpVal.d3;

				float tripletMinD = trpVal.minDistance(),
					tripletMaxD = trpVal.maxDistance(),
					tripletMinPhi = trpVal.minOrientation(),
					tripletMaxPhi = trpVal.maxOrientation();
				if (tripletMinD < minDTmp) { minDTmp = tripletMinD; }
				if (tripletMaxD > maxDTmp) { maxDTmp = tripletMaxD; }
				if (tripletMinPhi < minPhiTmp) { minPhiTmp = tripletMinPhi; }
				if (tripletMaxPhi > maxPhiTmp) { maxPhiTmp = tripletMaxPhi; }


				/*std::printf("Folder: %d, Template: %d (%d), Triplet %d\n", f, tpl, trp);
				std::printf("d1: %4.2f, d2: %4.2f, d3: %4.2f, phi1: %4.2f, phi2: %4.2f, phi3: %4.2f\n",
					trpVal.d1, trpVal.d2, trpVal.d3, trpVal.phi1, trpVal.phi2, trpVal.phi3);
				std::printf("Min d: %4.2f, phi %4.2f   Max: d: %4.2f, phi: %4.2f\n\n", tripletMinD, tripletMinPhi, tripletMaxD, tripletMaxPhi);*/
				//visualizeTripletOnEdges((*listTmpPtr)[tpl], triplets[trp], &trpVal);
			}
		}
	}
	TimeMeasuring tm(true);
	std::sort(distances.begin(), distances.end());
	std::printf("Sorted in %d[ms] - length: %d\n", tm.getTimeFromBeginning(), distances.size());
	std::printf("quantile 0.25 = %f\nquantile 0.5 = %f\nquantile 0.75 = %f\n\n", distances[(int)(distances.size() * 0.25)],
		distances[(int)(distances.size() * 0.5)], distances[(int)(distances.size() * 0.75)]);

	if (minD != NULL) { *minD = minDTmp; }
	if (maxD != NULL) { *maxD = maxDTmp; }
	if (minPhi != NULL) { *minPhi = minPhiTmp; }
	if (maxPhi != NULL) { *maxPhi = maxPhiTmp; }
}

HashSettings fillHashTable(TemplateHashTable &hashTable, FolderTemplateList &templates, int templatesLoaded, std::vector<Triplet> &triplets, int dBins, int phiBins) {
	std::vector<TripletValues> tripletsValues;
	float minD, maxD, minPhi, maxPhi;
	TimeMeasuring tm(true);
	countTripletsValues(tripletsValues, templates, triplets, templatesLoaded, &minD, &maxD, &minPhi, &maxPhi);
	std::printf("Count triplet vals in %d[ms]\n", tm.getTimeFromBeginning());
	HashSettings hashSettings(minD, maxD, minPhi, maxPhi, dBins, phiBins);

	std::printf("\nTripletsValues: %d\n\n", tripletsValues.size());
	std::printf("Min d: %4.2f, phi %4.2f   Max: d: %4.2f, phi: %4.2f\n\n", minD, minPhi, maxD, maxPhi);

	int const bins = 6;
	int dBinsCnt[bins] = { 0,0,0,0,0,0 };
	int phiBinsCnt[bins] = { 0,0,0,0,0,0 };

	for (int i = 0; i < tripletsValues.size(); i++)
	{
		QuantizedTripletValues hashKey(tripletsValues[i].tripletIndex,
			hashSettings.getDistanceBin(tripletsValues[i].d1),
			hashSettings.getDistanceBin(tripletsValues[i].d2),
			hashSettings.getDistanceBin(tripletsValues[i].d3),
			hashSettings.getOrientationBin(tripletsValues[i].phi1),
			hashSettings.getOrientationBin(tripletsValues[i].phi2),
			hashSettings.getOrientationBin(tripletsValues[i].phi3));
		hashTable[hashKey].push_back(tripletsValues[i].templateIndex);


		dBinsCnt[hashKey.d1]++;
		dBinsCnt[hashKey.d2]++;
		dBinsCnt[hashKey.d3]++;
		phiBinsCnt[hashKey.phi1]++;
		phiBinsCnt[hashKey.phi2]++;
		phiBinsCnt[hashKey.phi3]++;
	}


	std::printf("Distance bins:\n");
	for (int i = 0; i < bins; i++)
	{
		std::printf("\t%d:%8d\n", i, dBinsCnt[i]);
	}
	std::printf("\nOrientation bins:\n");
	for (int i = 0; i < bins; i++)
	{
		std::printf("\t%d:%8d\n", i, phiBinsCnt[i]);
	}
	std::printf("\n");

	return hashSettings;
}

int main()
{
	srand(time(0));
	TimeMeasuring elapsedTime(true);

	FolderTemplateList templates;
	int templatesLoaded = loadAllTemplates(templates);
	elapsedTime.insertBreakpoint("tplLoaded");
	std::printf("Templates loaded in: %d [ms]\n", elapsedTime.getTimeFromBeginning());

	std::vector<Triplet> triplets = generateTriplets(tripletsAmount, pointsInRowCol, pointsEdgeOffset, pointsDistance);
	elapsedTime.insertBreakpoint("genTriplets");
	std::printf("Triplets generated in: %d [us] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("tplLoaded",true), elapsedTime.getTimeFromBeginning()); 	
	//visualizeTriplets(triplets, pointsEdgeOffset, pointsDistance, 48);
	
	TemplateHashTable hashTable;
	HashSettings hashSettings = fillHashTable(hashTable, templates, templatesLoaded, triplets, distanceBins, orientationBins);
	//std::getc(stdin);
	elapsedTime.insertBreakpoint("hashTable");
	std::printf("Hash table filled in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("genTriplets"), elapsedTime.getTimeFromBeginning());
	std::printf("hash table size: %d\n", hashTable.size());
	int maxSize = 0;
	for (auto it = hashTable.begin(); it != hashTable.end(); ++it) {
		if (it->second.size() > maxSize)
		{
			maxSize = it->second.size();
		}
		/*if (it->second.size() > 100)
		{
			QuantizedTripletValues qtz = it->first;
			std::printf("%d = TrpI: %d _ D: %d, %d, %d _ Phi %d, %d, %d\n", it->second.size(),
				qtz.tripletIndex, qtz.d1, qtz.d2, qtz.d3, qtz.phi1, qtz.phi2, qtz.phi3);
			/*for (int i = 0; i < 100; i++)
			{
				visualizeTripletOnEdges(templates[it->second[i].folderIndex][it->second[i].templateIndex], triplets[qtz.tripletIndex], NULL, 300);
			}
			cv::waitKey();*/
		//}
	}
	std::printf("Max size: %d\n\n", maxSize);
	
	//maxSize++;
	//int *sizeCnt = new int[maxSize];
	//for (int f = 0; f < maxSize; f++)
	//{
	//	sizeCnt[f] = 0;
	//}
	//for (auto it = hashTable.begin(); it != hashTable.end(); ++it) {
	//	sizeCnt[it->second.size()]++;
	//}
	//for (int f = 0; f < maxSize; f++)
	//{
	//	if (sizeCnt[f] > 0)
	//	{
	//		std::printf("%d templates under %d keys\n", f, sizeCnt[f]);
	//		//std::getc(stdin);
	//	}
	//}	
	//delete sizeCnt;

	// paralel je jiz uvnitr funkce filterTemplateEdges
	// #pragma omp parallel for
	for (int f = 0; f < templates.size(); f++)
	{
		std::printf("Start %d - ", f);
		TimeMeasuring elapsedTime;
		elapsedTime.startMeasuring();
		//showResized("before", templates[f][0].edges_8u, 3);
		filterTemplateEdges(templates[f], kTpl, lambda, thetaD, thetaPhi, tau, removePixelRatio);
		//showResized("after", templates[f][0].edges_8u, 3);
		//cv::waitKey();
		std::printf("Folder %d of templates filtered in: %d [ms]\n", f, elapsedTime.getTimeFromBeginning());
	}
	elapsedTime.insertBreakpoint("filterEdges");
	std::printf("Edges filtered in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("hashTable"), elapsedTime.getTimeFromBeginning());

	std::printf("Total time: %d [ms]\n", elapsedTime.getTimeFromBeginning());
	
	std::getc(stdin);
	return 0;
	
	//return testDetectedEdgesAndDistanceTransform();

	//return filterTemplateEdges();
	//return selectingByStabilityToViewpoint();
	//return selectingByEdgeOrientations();

    return 0;
}
