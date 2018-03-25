#pragma once

#include <vector>

class HashSettings
{
public:
	float minD, maxD, minPhi, maxPhi;
	int dBins, phiBins;
	float dNormDivider, phiNormDivider;
	std::vector<float> dBinsRange;

	HashSettings() {}

	HashSettings(float minD, float maxD, float minPhi, float maxPhi, int dBins, int phiBins) {
		this->minD = minD;
		this->maxD = maxD;
		this->minPhi = minPhi;
		this->maxPhi = maxPhi;

		this->dBins = dBins;
		this->phiBins = phiBins;

		this->dNormDivider = (maxD - minD) / (float)dBins;
		this->phiNormDivider = (maxPhi - minPhi) / (float)phiBins;
	}

	int getDistanceBin(float distance) {
		if (this->dBinsRange.size() > 0)
		{
			int i = 0;
			for (; i < this->dBinsRange.size(); i++)
			{
				if (distance <= this->dBinsRange[i]) {
					return i;
				}
			}
			return i;
		}
		return 4;
		/*if (distance < 1.2f)
		{
			return 0;
		}
		if (distance < 3.4f)
		{
			return 1;
		}
		if (distance < 6.5f)
		{
			return 2;
		}
		return 3;

		/*int bin = (distance - this->minD) / this->dNormDivider;
		if (bin >= this->dBins)
		{
			return this->dBins - 1;
		}
		return bin;*/
	}

	int getOrientationBin(float orientation) {
		int bin = (orientation - this->minPhi) / this->phiNormDivider;
		if (bin >= this->phiBins)
		{
			return this->phiBins - 1;
		}
		return bin;
	}
};

