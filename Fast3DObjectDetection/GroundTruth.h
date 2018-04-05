#pragma once

#include "constantsAndTypes.h"

class GroundTruth
{
public:
	cv::Rect rect;
	int folderIndex;
	GroundTruth() {};

	GroundTruth(cv::Rect rect, int folderIndex){
		this->rect = rect;
		this->folderIndex = folderIndex;
	}

	GroundTruth(int xTL, int yTL, int xBR, int yBR, int folderIndex){
		this->rect = cv::Rect(cv::Point(xTL, yTL), cv::Point(xBR, yBR));
		this->folderIndex = folderIndex;
	}

	float percentageOverlap(const GroundTruth &c2) {
		return (float)((this->rect & c2.rect).area()) / (float)(std::min(this->rect.area(), c2.rect.area()));
	}

	float intersectOverUnion(const GroundTruth &c2) {
		int intersectArea = (this->rect & c2.rect).area();
		return (float)intersectArea / (float)(this->rect.area() + c2.rect.area() - intersectArea);
	}

	bool operator < (const GroundTruth& c2) const
	{
		if (this->rect.x == c2.rect.x)
		{
			return (this->rect.y < c2.rect.y);
		}
		return (this->rect.x < c2.rect.x);
	}
};
