#include "constantsAndTypes.h"
#include "distanceAndOrientation.h"

#include <iomanip>

// --------- edges + distance transform -----------
cv::Mat getDetectedEdges_8u(cv::Mat &src_8u) {
	cv::Mat detected_edges_8u, dst;

	// Image src_8u is already blurred
	// Canny detector - need 8U matrix
	cv::Canny(src_8u, detected_edges_8u, canny_LowThreshold, canny_LowThreshold*canny_Ratio, canny_KernelSize);

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

// --------- edge distance and orientation -----------
float getEdgeOrientation(cv::Mat &srcGray_8u, int x, int y, bool onlyPositive) {
	int xStart = (x == 0 ? 1 : 0), // If edge is calculating - skip part of mask
		xEnd = (x == srcGray_8u.cols - 1) ? 2 : 3, // If is edge on end of picture
		yStart = (y == 0 ? 1 : 0),
		yEnd = (y == srcGray_8u.rows - 1) ? 2 : 3;

	float sXVal = 0.0f;
	float sYVal = 0.0f;
	for (int mX = xStart; mX < xEnd; mX++)
	{
		for (int mY = yStart; mY < yEnd; mY++)
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
	int substract = (x > 0 && y > 0) ? 1 : -1;
	float pointDistance = distTrans_32f.at<float>(y, x);
	float angle = atan2(pointDistance - (float)distTrans_32f.at<float>(y - substract, x), pointDistance - (float)distTrans_32f.at<float>(y, x - substract));
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


