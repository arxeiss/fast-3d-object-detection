#include "sparseDescriptor.h"
#include "distAndOri.h"

std::vector<float> getImgVector(cv::Mat img_32f, int step, int margin, int startX, int startY, int width, int height) {
	if (width < 0) {
		width = img_32f.cols;
	}
	if (height < 0) {
		height = img_32f.rows;
	}
	// +1 for adding initial point
	int halfOfVector = (int)((width - 2 * margin) / step + 1) *
		(int)((height - 2 * margin) / step + 1);

	std::vector<float> imgVector(halfOfVector * 2, 0.0f);

	int vectorPos = 0;
	for (int x = margin + startX; x <= width - margin + startX; x += step) {
		for (int y = margin + startY; y <= height - margin + startY; y += step) {
			float pointDistance = img_32f.at<float>(y, x);
			float angle = atan2(pointDistance - img_32f.at<float>(y - 1, x), pointDistance - img_32f.at<float>(y, x - 1));
			imgVector[vectorPos] = pointDistance;
			imgVector[vectorPos + halfOfVector] = angle;
			//cv::circle(img, cv::Point(x, y), 1, cv::Scalar(1));
			//std::printf("Put to index [%d %d]\n", vectorPos, vectorPos + halfOfVector);
			//int dX = x - round(cos(angle) * 6);
			//int dY = y - round(sin(angle) * 6);
			//cv::line(img, cv::Point(x, y), cv::Point(dX, dY), cv::Scalar(1));
			vectorPos++;
		}
	}

	return imgVector;
}

int compareVectors(std::vector<float> v1, std::vector<float> v2, float thetaD, float thetaPhi) {
	int halfOfVector = std::min(v1.size(), v2.size()) / 2;

	int simmilarity = 0;
	for (int i = 0; i < halfOfVector; i++)
	{
		float deltaDistance = std::abs(v1[i] - v2[i]);
		float deltaAngle = std::abs(v1[i + halfOfVector] - v2[i + halfOfVector]);

		if (deltaDistance < thetaD) {
			simmilarity++;
		}
		if (deltaAngle < thetaPhi) {
			simmilarity++;
		}
	}

	return simmilarity;
}

void runSparseDescriptor(float thetaD, float thetaPhi) {
	cv::Mat src = cv::imread("images/lena.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat srcDistanceTransform_32f = getDistanceTransform_32f(src);

	cv::Mat search = cv::imread("images/search.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat searchDistanceTransform_32f = getDistanceTransform_32f(search);


	int margin = 5;
	int step = 10;

	std::vector<float> searchVector = getImgVector(searchDistanceTransform_32f, step, margin);

	int bestSimmilarity = 0;
	int bestX = 0, bestY = 0;

	for (int x = 0; x < src.cols - search.cols; x += step) {
		for (int y = 0; y < src.rows - search.rows; y += step) {

			std::vector<float> currentVector = getImgVector(srcDistanceTransform_32f, step, margin, x, y, search.cols, search.rows);
			int simmilarity = compareVectors(searchVector, currentVector, thetaD, thetaPhi);

			if (simmilarity > bestSimmilarity) {
				std::printf("%d:%d - %d\n", x, y, simmilarity);
				bestSimmilarity = simmilarity;
				bestX = x;
				bestY = y;
			}
		}
	}

	std::printf("X, Y = [%d %d]", bestX, bestY);

	for (int x = 0; x < search.cols; x++)
	{
		for (int y = 0; y < search.rows; y++)
		{
			src.at<uchar>(y + bestY, x + bestX) = src.at<uchar>(y + bestY, x + bestX) * 0.3 + search.at<uchar>(y, x) * 0.7;
		}
	}

	//cv::rectangle(src, cv::Rect(bestX, bestY, search.cols, search.rows), cv::Scalar(200));

	cv::imshow("edges", src);
	cv::waitKey(0);
}
