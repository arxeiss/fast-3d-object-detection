#include "constants.h"
#include "distAndOri.h"

#include <iomanip>

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
float getEdgeOrientation(cv::Mat &srcGray_8u, int x, int y, bool onlyPositive) {
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

float getEdgeOrientationFromDistanceTransform(cv::Mat &distTrans_32f, int x, int y, bool onlyPositive) {
	float pointDistance = distTrans_32f.at<float>(y, x);
	float angle = atan2(pointDistance - (float)distTrans_32f.at<float>(y - 1, x), pointDistance - (float)distTrans_32f.at<float>(y, x - 1));
	if (angle < 0 && onlyPositive)
	{
		angle += M_PI;
	}
	return angle;
}

void getEdgeDistAndOri(DetectionUnit &img, int x, int y, float &distance, float &orientation, bool onlyPositive) {
	distance = img.distanceTransform_32f.at<float>(y, x);
	orientation = (distance == 0.0f) ?
		getEdgeOrientation(img.img_8u, x, y, onlyPositive) :
		getEdgeOrientationFromDistanceTransform(img.distanceTransform_32f, x, y, onlyPositive);
}

/// TEST func -- potrebuje edges+distance transform; edge distance and orientation; 
void showEdgeOrientations(cv::Mat src_8u, cv::Mat edge_8u, bool fromDistanceTransform, std::string windowName) {
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
