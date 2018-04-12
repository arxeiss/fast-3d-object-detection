#pragma once

#include "GroundTruth.h"
#include "TemplateIndex.h"

class Candidate : public GroundTruth
{
public:
	TemplateIndex tplIndex;
	float chamferScore;

	Candidate() :tplIndex(0,0) {}

	Candidate(cv::Rect rect, TemplateIndex tplIndex, float chamferScore, bool active = true) : tplIndex(tplIndex) {
		this->folderIndex = tplIndex.folderIndex;
		this->rect = rect;
		this->chamferScore = chamferScore;
		this->active = active;
	}

	Candidate(int x, int y, int sideLength, TemplateIndex tplIndex, float chamferScore, bool active = true) : tplIndex(tplIndex) {
		this->folderIndex = tplIndex.folderIndex;
		this->rect = cv::Rect(x, y, sideLength, sideLength);
		this->chamferScore = chamferScore;
		this->active = active;
	}
};
