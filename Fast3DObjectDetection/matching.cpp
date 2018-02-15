#include "matching.h"

#include "visualize.h"
#include "loading.h"

void matchInImage(cv::Mat testImg_8u, HashSettings &hashSettings, std::vector<Triplet> triplets, TemplateHashTable hashTable) {
	//int slidingWindowSize = 150;
	int maxX = testImg_8u.cols - slidingWindowSize + 1; 
	int maxY = testImg_8u.rows - slidingWindowSize + 1;
	cv::Mat toDraw;
	testImg_8u.copyTo(toDraw);
	for (int x = 0; x < maxX; x += slidingWindowStep)
	{
		for (int y = 0; y < maxY; y += slidingWindowStep)
		{
			DetectionUnit unit = getDetectionUnitByROI(testImg_8u, x, y, slidingWindowSize);
			if (unit.edgesCount == 0) {
				continue;
			}

			std::unordered_map<TemplateIndex, int> candidatesCount;
			for (int i = 0; i < triplets.size(); i++) {
				QuantizedTripletValues hashKey = getTableHashKey(hashSettings, unit, triplets[i], i);
				//std::printf("Hashtable - Cnt: %d\n", hashTable.count(hashKey));
				if (hashTable.count(hashKey)) {
					// Pristupuji pod klic ktery neexistuje!!! k je z tripletu, a ne z hashtable
					//std::printf("Triplet %d - candidates %d\n", i, hashTable[hashKey].size());
					for (int k = 0; k < hashTable[hashKey].size(); k++)
					{
						int c = candidatesCount[hashTable[hashKey][k]]++;
						//std::printf("\t%2d: %2d/%4d - %dx\n", k, hashTable[hashKey][k].folderIndex, hashTable[hashKey][k].templateIndex, c);
						if (c > 0) {
							//std::printf("Here!!! %2d: %2d/%4d - %dx\n", k, hashTable[hashKey][k].folderIndex, hashTable[hashKey][k].templateIndex, c);
							//std::printf("Here!!!");
						}
					}
				}
			}
			int moreTimesThanThetaV = 0;
			for (auto it = candidatesCount.begin(); it != candidatesCount.end(); it++) {
				if (it->second >= thetaV) {
					moreTimesThanThetaV++;
				}
			}
			int delay = 1;
			if (moreTimesThanThetaV > 0)
			{
				std::printf("Total candidates %d - more than 3: %d\n", candidatesCount.size(), moreTimesThanThetaV);
				delay = 0;
				drawSlidingWindowToImage(toDraw, slidingWindowSize, x, y);
			}	

			//showSlidingWindowInImage(testImg_8u, slidingWindowSize, x, y, delay);
			/*int key = showDetectionUnit(unit);
			if (key == 'h') { x += 60; y = 0; }
			if (key == 'v') { y += 60; }
			if (key == 'c') { return; }*/
		}
	}
	cv::imshow("Possible candidates", toDraw);
	cv::waitKey();
}