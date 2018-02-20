#include "matching.h"

#include "visualize.h"
#include "utils.h"
#include "loading.h"
#include "TimeMeasuring.h"
#include "chamferScore.h"

void matchInImage(cv::Mat &testImg_8u, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges) {
	
	cv::Mat toDraw;
	testImg_8u.copyTo(toDraw);

	TimeMeasuring tm;
	tm.startMeasuring();

	for (int i = 9; i >= 0; i--)
	{
		cv::Mat sizedScene;
		printf("\n\n## SCALE PYRAMID - step %d ##\n", i);
		float scaleRatio = fastPow(scalePyramidStep, i);
		cv::resize(testImg_8u, sizedScene, cv::Size(round(testImg_8u.cols / scaleRatio), round(testImg_8u.rows / scaleRatio)));		
		matchInImageWithSlidingWindow(sizedScene, toDraw, templates, hashSettings, triplets, hashTable, averageEdges, scaleRatio);
		cv::imshow("Possible candidates", toDraw);
		cv::waitKey();
	}
	std::printf("\n\n\nTotal matching time: %d [ms]", tm.getTimeFromBeginning());
	cv::imshow("Possible candidates", toDraw);
	cv::waitKey();
}

void matchInImageWithSlidingWindow(cv::Mat &scene_8u, cv::Mat &previewScene, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, float sceneScaleRatio) {
	int maxX = scene_8u.cols - slidingWindowSize + 1;
	int maxY = scene_8u.rows - slidingWindowSize + 1;

	TimeMeasuring tm;
	tm.startMeasuring();

#pragma omp parallel for
	for (int x = 0; x < maxX; x += slidingWindowStep)
	{
		for (int y = 0; y < maxY; y += slidingWindowStep)
		{
			DetectionUnit unit = getDetectionUnitByROI(scene_8u, x, y, slidingWindowSize);
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
						//if (c > 0) {
						//std::printf("Here!!! %2d: %2d/%4d - %dx\n", k, hashTable[hashKey][k].folderIndex, hashTable[hashKey][k].templateIndex, c);
						//std::printf("Here!!!");
						//}
					}
				}
			}
			int moreTimesThanThetaV = 0;
			float bestChamferScore = -1;
			TemplateIndex bestTemplateIndex(-1, -1);
			for (auto it = candidatesCount.begin(); it != candidatesCount.end(); it++) {
				if (it->second >= thetaV) {
					moreTimesThanThetaV++;
					float score = getOrientedChamferScore(templates[it->first.folderIndex][it->first.templateIndex], unit,
						averageEdges, lambda, thetaD, thetaPhi);
					if (score > bestChamferScore) {
						bestChamferScore = score;
						bestTemplateIndex = it->first;
					}
				}
			}
			if (moreTimesThanThetaV > 0 /*&& bestChamferScore >= 0.05*/)
			{
				std::printf("Total candidates %4d - more than 3: %2d, best score: %4.5f\n", candidatesCount.size(), moreTimesThanThetaV, bestChamferScore);
#pragma omp critical
				drawSlidingWindowToImage(previewScene, slidingWindowSize * sceneScaleRatio, x * sceneScaleRatio, y * sceneScaleRatio, bestChamferScore * 4);
				//cv::imshow("Possible candidates", toDraw);
				//cv::waitKey();
			}

			//showSlidingWindowInImage(scene_8u, slidingWindowSize, x, y, delay);
			/*int key = showDetectionUnit(unit);
			if (key == 'h') { x += 60; y = 0; }
			if (key == 'v') { y += 60; }
			if (key == 'c') { return; }*/
		}
	}
	std::printf("Matching in %d [ms]", tm.getTimeFromBeginning());
}