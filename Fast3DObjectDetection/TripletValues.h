#pragma once
//#include "TemplateIndex.h"
class TripletValues
{
public:
	float d1, d2, d3, phi1, phi2, phi3;
	int tripletIndex = 0;
	TemplateIndex templateIndex;

	TripletValues(): templateIndex(0, 0){ }

	TripletValues(const int tripletIndex, const TemplateIndex &templateIndex,
		const float d1 = 0.0f, const float d2 = 0.0f, const float d3 = 0.0f,
		const float phi1 = 0.0f, const float phi2 = 0.0f, const float phi3 = 0.0f) : templateIndex(templateIndex)
	{		
		this->d1 = d1;
		this->d2 = d2;
		this->d3 = d3;
		
		this->phi1 = phi1;
		this->phi2 = phi2;
		this->phi3 = phi3;

		this->tripletIndex = tripletIndex;
	}

	float minDistance() {
		return std::min({ this->d1, this->d2, this->d3 });
	}
	float maxDistance() {
		return std::max({ this->d1, this->d2, this->d3 });
	}
	float minOrientation() {
		return std::min({ this->phi1, this->phi2, this->phi3 });
	}
	float maxOrientation() {
		return std::max({ this->phi1, this->phi2, this->phi3 });
	}
};

