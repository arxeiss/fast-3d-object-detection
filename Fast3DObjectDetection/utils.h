#pragma once

inline void printMatType(cv::Mat &img) {
	int type = img.type();
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	printf("Matrix: %s %dx%d \n", r, img.cols, img.rows);
}

inline double fastPow(double base, unsigned int exp) {
	if (exp < 1) {
		return 1.0f;
	}
	double result = base;
	while (--exp > 0) {
		result *= base;
	}
	return result;
}

// Thanks to https://github.com/kobzol and https://stackoverflow.com/a/12996028/1107768
inline unsigned int improveIntHash(unsigned int x) {
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = ((x >> 16) ^ x) * 0x45d9f3b;
	x = (x >> 16) ^ x;
	return x;
}