#pragma once
class TemplateChamferScore
{
public:
	int index;
	double chamferScore;

	TemplateChamferScore() {};
	TemplateChamferScore(int index, double chamferScore) {
		this->index = index;
		this->chamferScore = chamferScore;
	};
	//~TemplateChamferScore();
};

