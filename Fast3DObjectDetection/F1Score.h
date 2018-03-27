#pragma once

class F1Score {
public:
	int truePositive,
		falsePositive,
		falseNegative,
		trueNegative;
	F1Score() :F1Score(0, 0, 0, 0) {}
	F1Score(int truePositive, int falsePositive, int falseNegative) :
		F1Score(truePositive, falsePositive, falseNegative, 0) {}
	F1Score(int truePositive, int falsePositive, int falseNegative, int trueNegative) {
		this->truePositive = truePositive;
		this->falsePositive = falsePositive;
		this->falseNegative = falseNegative;
		this->trueNegative = trueNegative;
	}

	float getPrecision() {
		return (float)this->truePositive / (float)(this->truePositive + this->falsePositive);
	}

	float getRecall() {
		return (float)this->truePositive / (float)(this->truePositive + this->falseNegative);
	}

	float getF1Score(bool percentage = false) {
		float score = 2.0f * (this->getPrecision() * this->getRecall()) / (this->getPrecision() + this->getRecall());
		if (percentage)
		{
			score *= 100;
		}
		return score;
	}

	F1Score operator+(const F1Score& add) {
		F1Score score(*this);

		score.truePositive += add.truePositive;
		score.falsePositive += add.falsePositive;
		score.falseNegative += add.falseNegative;
		score.trueNegative += add.trueNegative;

		return score;
	}

	F1Score& operator+=(const F1Score& add) {
		this->truePositive += add.truePositive;
		this->falsePositive += add.falsePositive;
		this->falseNegative += add.falseNegative;
		this->trueNegative += add.trueNegative;

		return *this;
	}
};