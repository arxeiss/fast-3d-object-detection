#pragma once

#include "constantsAndTypes.h"
#include "TemplateIndex.h"

class Candidate
{
public:
	cv::Rect rect;
	TemplateIndex tplIndex;
	float chamferScore;
	bool active = false;

	Candidate() :tplIndex(0,0) {}

	Candidate(cv::Rect rect, TemplateIndex tplIndex, float chamferScore, bool active = true) : tplIndex(tplIndex) {
		this->rect = rect;
		this->chamferScore = chamferScore;
		this->active = active;
	}

	Candidate(int x, int y, int sideLength, TemplateIndex tplIndex, float chamferScore, bool active = true) : tplIndex(tplIndex) {
		this->rect = cv::Rect(x, y, sideLength, sideLength);
		this->chamferScore = chamferScore;
		this->active = active;
	}

	bool operator < (const Candidate& c2) const
	{
		if (this->rect.x == c2.rect.x)
		{
			return (this->rect.y < c2.rect.y);
		}
		return (this->rect.x < c2.rect.x);
	}

	float percentageOverlap(const Candidate &c2) {
		return (float)((this->rect & c2.rect).area()) / (float)(std::min(this->rect.area(), c2.rect.area()));
	}

	//~Candidate();
};
