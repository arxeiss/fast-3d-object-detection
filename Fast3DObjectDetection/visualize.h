#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iomanip>

#include "DetectionUnit.h"
#include "Triplet.h"
#include "TripletValues.h"

#include "distanceAndOrientation.h"

inline void showResized(cv::String label, cv::Mat img, int ratio, int waitToDraw = -1) {
	cv::Mat toShow;
	if (ratio > 1)
	{
		cv::resize(img, toShow, cv::Size(), ratio, ratio, cv::INTER_AREA);
	}
	else
	{
		toShow = img;
	}
	cv::imshow(label, toShow);
	if (waitToDraw >= 0) {
		cv::waitKey(waitToDraw);
	}
}

// Not used anymore
inline void drawEdgesToSource(cv::Mat &src_8u3c, cv::Mat &edges_8u, int xOffset, int yOffset, float scaleRatio, float edgeRatio = 0.99) {
	
	for (int x = 0; x < edges_8u.cols; x++)
	{
		for (int y = 0; y < edges_8u.rows; y++)
		{
			if (edges_8u.at<uchar>(y, x) == 0) {
				int rX = x * scaleRatio + xOffset, rY = y * scaleRatio + yOffset;
				src_8u3c.at<cv::Vec3b>(rY, rX) = src_8u3c.at<cv::Vec3b>(rY, rX) * (1 - edgeRatio) + cv::Vec3b(0, 255, 0) * edgeRatio;
			}
		}
	}
}

inline int showDetectionUnit(DetectionUnit &unit, int delay = 0, std::string windowName = "Detection unit") {
	cv::Mat ret;
	unit.img_8u.copyTo(ret);
	cv::cvtColor(ret, ret, CV_GRAY2BGR);
	for (int x = 0; x < ret.cols; x++)
	{
		for (int y = 0; y < ret.rows; y++)
		{
			if (unit.edges_8u.at<uchar>(y, x) == 0)
			{
				ret.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
			}
		}
	}
	cv::imshow(windowName, ret);
	return cv::waitKey(delay);
}

inline void drawSlidingWindowToImage(cv::Mat &mat, int windowSize, int windowX, int windowY, float colorMultiply = 1.0f, std::string str = "") {
	if (mat.channels() == 1)
	{
		cv::cvtColor(mat, mat, CV_GRAY2BGR);
	}
	cv::rectangle(mat, cv::Rect(windowX, windowY, windowSize, windowSize), cv::Scalar(50, 50, 255 * colorMultiply));
	if (str.length())
	{
		cv::putText(mat, str, cv::Point(windowX, windowY - 5), CV_FONT_HERSHEY_SIMPLEX, 0.35f, cv::Scalar(30, 200, 30), 1);
	}
}

inline int showSlidingWindowInImage(cv::Mat &img, int windowSize, int windowX, int windowY, int delay = 0) {
	cv::Mat ret;
	img.copyTo(ret);
	drawSlidingWindowToImage(ret, windowSize, windowX, windowY);
	cv::imshow("Sliding window", ret);
	return cv::waitKey(delay);
}

/// TEST func
inline int visualizeTriplets(std::vector<Triplet> &triplets, int edgeOffset, int pointsDistance, int imageColsRows, int scaleRatio = 10, int wait = 100)
{
	cv::Mat netRaw, netSingle, netAll;

	netRaw = cv::Mat(imageColsRows * scaleRatio, imageColsRows * scaleRatio, CV_8UC3);
	netRaw = cv::Scalar(55, 55, 55);
	for (int x = edgeOffset; x < imageColsRows; x += pointsDistance)
	{
		cv::line(netRaw, cv::Point(x * scaleRatio, 0), cv::Point(x * scaleRatio, netRaw.cols), cv::Scalar(128, 128, 128));
		cv::line(netRaw, cv::Point(0, x * scaleRatio), cv::Point(netRaw.cols, x * scaleRatio), cv::Scalar(128, 128, 128));
	}
	for (int x = edgeOffset; x < imageColsRows; x += pointsDistance)
	{
		for (int y = edgeOffset; y < imageColsRows; y += pointsDistance)
		{
			cv::circle(netRaw, cv::Point(x * scaleRatio, y * scaleRatio), 4, cv::Scalar(0, 0, 0), -1);
		}
	}
	netRaw.copyTo(netSingle);
	netRaw.copyTo(netAll);
	cv::imshow("netAll", netAll);
	for (int i = 0; i < triplets.size(); i++)
	{
		cv::line(netAll, triplets[i].p1 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
		cv::line(netAll, triplets[i].p3 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
		cv::circle(netAll, triplets[i].p1 * scaleRatio, 4, cv::Scalar(255, 255, 255), -1);
		cv::circle(netAll, triplets[i].p2 * scaleRatio, 4, cv::Scalar(255, 255, 255), -1);
		cv::circle(netAll, triplets[i].p3 * scaleRatio, 4, cv::Scalar(255, 255, 255), -1);

		cv::line(netSingle, triplets[i].p1 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
		cv::line(netSingle, triplets[i].p3 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
		cv::circle(netSingle, triplets[i].p1 * scaleRatio, 4, cv::Scalar(0, 0, 200), -1);
		cv::circle(netSingle, triplets[i].p2 * scaleRatio, 4, cv::Scalar(0, 200, 0), -1);
		cv::circle(netSingle, triplets[i].p3 * scaleRatio, 4, cv::Scalar(200, 0, 0), -1);

		cv::imshow("netSingle", netSingle);
		cv::waitKey(wait);
		cv::imshow("netAll", netAll);

		for (int x = 0; x < netRaw.cols; x++)
		{
			for (int y = 0; y < netRaw.rows; y++)
			{
				netSingle.at<cv::Vec3b>(y, x) = netRaw.at<cv::Vec3b>(y, x);
			}
		}
	}
	cv::waitKey();
	return 0;
}

/// TEST func
inline void visualizeTripletOnEdges(DetectionUnit &unit, Triplet &triplet, TripletValues *tripletValues = NULL, int wait = 0) {
	int scaleRatio = 10;
	cv::Mat show;
	cv::resize(unit.edges_8u, show, cv::Size(), scaleRatio, scaleRatio, cv::INTER_AREA);
	for (size_t i = 0; i < show.cols * show.rows; i++)
	{
		if (show.at<uchar>(i) == 0)
		{
			show.at<uchar>(i) = 180;
		}
	}
	cv::cvtColor(show, show, CV_GRAY2BGR);

	cv::Point p1 = triplet.p1 * scaleRatio;
	p1.x += scaleRatio / 2;
	p1.y += scaleRatio / 2;
	cv::Point p2 = triplet.p2 * scaleRatio;
	p2.x += scaleRatio / 2;
	p2.y += scaleRatio / 2;
	cv::Point p3 = triplet.p3 * scaleRatio;
	p3.x += scaleRatio / 2;
	p3.y += scaleRatio / 2;

	cv::line(show, p1, p2, cv::Scalar(0, 0, 255), 2);
	cv::line(show, p3, p2, cv::Scalar(0, 0, 255), 2);
	cv::circle(show, p1, 4, cv::Scalar(0, 0, 200), -1);
	cv::circle(show, p2, 4, cv::Scalar(0, 170, 0), -1);
	cv::circle(show, p3, 4, cv::Scalar(200, 0, 0), -1);
	if (tripletValues != NULL)
	{
		int oriLineLength = 30;
		cv::line(show, p1, cv::Point(p1.x - round(cos(tripletValues->phi1) * oriLineLength), p1.y - round(sin(tripletValues->phi1) * oriLineLength)), cv::Scalar(0, 178, 191), 2);
		cv::line(show, p2, cv::Point(p2.x - round(cos(tripletValues->phi2) * oriLineLength), p2.y - round(sin(tripletValues->phi2) * oriLineLength)), cv::Scalar(0, 178, 191), 2);
		cv::line(show, p3, cv::Point(p3.x - round(cos(tripletValues->phi3) * oriLineLength), p3.y - round(sin(tripletValues->phi3) * oriLineLength)), cv::Scalar(0, 178, 191), 2);

		int textLength = 140;
		p1.x += 1 * scaleRatio; p2.x += 1 * scaleRatio; p3.x += 1 * scaleRatio;
		p1.y -= 1 * scaleRatio; p2.y -= 1 * scaleRatio; p3.y -= 1 * scaleRatio;
		if (p1.x + textLength > show.cols) { p1.x -= textLength; }
		if (p2.x + textLength > show.cols) { p2.x -= textLength; }
		if (p3.x + textLength > show.cols) { p3.x -= textLength; }

		std::stringstream ss;
		ss << std::fixed << std::setprecision(2) << "d: " << tripletValues->d1 << " phi: " << tripletValues->phi1;
		cv::putText(show, ss.str(), p1, CV_FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0, 0, 200), 1);
		ss.str("");
		ss.clear();
		ss << std::fixed << std::setprecision(2) << "d: " << tripletValues->d2 << " phi: " << tripletValues->phi2;
		cv::putText(show, ss.str(), p2, CV_FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(0, 170, 0), 1);
		ss.str("");
		ss.clear();
		ss << std::fixed << std::setprecision(2) << "d: " << tripletValues->d3 << " phi: " << tripletValues->phi3;
		cv::putText(show, ss.str(), p3, CV_FONT_HERSHEY_SIMPLEX, 0.3f, cv::Scalar(200, 0, 0), 1);
	}

	cv::imshow("Triplet on edges", show);
	cv::waitKey(wait);
}

/// TEST func - potrebuje ty 2 vyse
inline int testDetectedEdgesAndDistanceTransform() {
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

//inline void showEdgeOrientations(cv::Mat src_8u, cv::Mat edge_8u, bool fromDistanceTransform = false, std::string windowName = "Orientations") {
//	cv::Mat edgeWithOri, distanceTransform_32f;
//	cv::cvtColor(edge_8u, edgeWithOri, CV_GRAY2BGR);
//
//	if (fromDistanceTransform) {
//		distanceTransform_32f = getDistanceTransformFromEdges_32f(edge_8u);
//	}
//
//	bool onlyPositive = false;
//
//	for (int y = 5; y < edge_8u.rows; y += 5)
//	{
//		for (int x = 0; x < edge_8u.cols; x++)
//		{
//			if (edge_8u.at<uchar>(y, x) > 0)
//			{
//				continue;
//			}
//			//cv::circle(edgeWithOri, cv::Point(x, y), 1, cv::Scalar(1));
//			if (fromDistanceTransform)
//			{
//				int offsetX = x - 6, offsetY = y - 6;
//				float angle = getEdgeOrientationFromDistanceTransform(distanceTransform_32f, offsetX, offsetY, onlyPositive);
//				int dX = offsetX - round(cos(angle) * 6);
//				int dY = offsetY - round(sin(angle) * 6);
//				cv::line(edgeWithOri, cv::Point(offsetX, offsetY), cv::Point(dX, dY), cv::Scalar(0, 0, 255));
//				cv::circle(edgeWithOri, cv::Point(offsetX, offsetY), 1, cv::Scalar(255, 0, 0), -1);
//			}
//			else {
//				float angle = getEdgeOrientation(src_8u, x, y, onlyPositive);
//				int dX = x - round(cos(angle) * 6);
//				int dY = y - round(sin(angle) * 6);
//				cv::line(edgeWithOri, cv::Point(x, y), cv::Point(dX, dY), cv::Scalar(0, 0, 255));
//			}
//			x += 4;
//		}
//	}
//
//	cv::imshow(windowName, edgeWithOri);
//}