#include "matching.h"

#include "visualize.h"
#include "utils.h"
#include "loading.h"
#include "TimeMeasuring.h"
#include "chamferScore.h"

void matchInImage(cv::Mat &testImg_8u, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges) {
	
	cv::Mat toDraw;

	TimeMeasuring tm;
	tm.startMeasuring();

	std::vector<Candidate> candidates;

	for (int i = 9; i >= 0; i--)
	//for (int i = 0; i < 9; i++)
	{
		cv::Mat sizedScene;
		printf("\n\n## SCALE PYRAMID - step %d ##\n", i);
		float scaleRatio = fastPow(scalePyramidStep, i);
		cv::resize(testImg_8u, sizedScene, cv::Size(round(testImg_8u.cols / scaleRatio), round(testImg_8u.rows / scaleRatio)));		
		matchInImageWithSlidingWindow(sizedScene, candidates, templates, hashSettings, triplets, hashTable, averageEdges, scaleRatio);

		/*testImg_8u.copyTo(toDraw);
		for (int i = 0; i < candidates.size(); i++)
		{
			drawSlidingWindowToImage(toDraw, candidates[i].rect.width, candidates[i].rect.x, candidates[i].rect.y, candidates[i].chamferScore * 4);
		}
		cv::imshow("Possible candidates", toDraw);
		cv::waitKey();*/
	}
	std::printf("\n\n\nTotal matching time: %d [ms]\n", tm.getTimeFromBeginning());
	testImg_8u.copyTo(toDraw);
	for (int i = 0; i < candidates.size(); i++)
	{
		drawSlidingWindowToImage(toDraw, candidates[i].rect.width, candidates[i].rect.x, candidates[i].rect.y, candidates[i].chamferScore * 4);
	}
	std::printf("Total windows: %d\n", candidates.size());
	cv::imshow("Possible candidates", toDraw);

	tm.insertBreakpoint("nms");
	std::sort(candidates.begin(), candidates.end());
	nonMaximaSupression(candidates);
	std::printf("NMS time: %d [us]\n", tm.getTimeFromBreakpoint("nms", true));
	
	cv::Mat NMS;
	testImg_8u.copyTo(NMS);
	int windows = 0;
	for (int i = 0; i < candidates.size(); i++)
	{
		if (candidates[i].active)
		{
			windows++;
			std::stringstream ss;
			ss << "F: " << candidates[i].tplIndex.folderIndex << " T: " << candidates[i].tplIndex.templateIndex;
			drawSlidingWindowToImage(NMS, candidates[i].rect.width, candidates[i].rect.x, candidates[i].rect.y, candidates[i].chamferScore * 4, ss.str());
			drawEdgesToSource(NMS, templates[candidates[i].tplIndex.folderIndex][candidates[i].tplIndex.templateIndex].edges_8u, candidates[i].rect.x, candidates[i].rect.y, (float)candidates[i].rect.width / (float)slidingWindowSize);
		}
	}
	std::printf("Total windows after NMS: %d\n", windows);
	cv::imshow("Detection", NMS);
	cv::waitKey();
}

void matchInImageWithSlidingWindow(cv::Mat &scene_8u, std::vector<Candidate> &candidates, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, float sceneScaleRatio) {
	int maxX = scene_8u.cols - slidingWindowSize + 1;
	int maxY = scene_8u.rows - slidingWindowSize + 1;

	TimeMeasuring tm;
	tm.startMeasuring();

#pragma omp parallel for
	for (int x = 0; x < maxX; x += slidingWindowStep)
	{
		for (int y = 0; y < maxY; y += slidingWindowStep)
		{
			Candidate candidate = computeMatchInSlidingWindow(scene_8u, x, y, templates, hashSettings, triplets, hashTable, averageEdges, sceneScaleRatio);
			if (candidate.active)
			{			
				#pragma omp critical
				candidates.push_back(candidate);
			}
		}
	}
	std::printf("Matching in %d [ms]", tm.getTimeFromBeginning());
}

Candidate computeMatchInSlidingWindow(cv::Mat &scene_8u, int x, int y, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, float sceneScaleRatio) {
	DetectionUnit unit = getDetectionUnitByROI(scene_8u, x, y, slidingWindowSize);
	if (unit.edgesCount == 0) {
		return Candidate();
	}

	tsl::sparse_map<TemplateIndex, int> candidatesCount;
	for (int i = 0; i < triplets.size(); i++) {
		QuantizedTripletValues hashKey = getTableHashKey(hashSettings, unit, triplets[i], i);
		if (hashTable.count(hashKey)) {
			for (int k = 0; k < hashTable[hashKey].size(); k++)
			{
				candidatesCount[hashTable[hashKey][k]]++;
			}
		}
	}
	int moreTimesThanThetaV = 0;
	float bestChamferScore = -1;
	TemplateIndex bestTemplateIndex(-1, -1);
	for (auto it = candidatesCount.begin(); it != candidatesCount.end(); it++) {
		if (it->second >= thetaV) {
			moreTimesThanThetaV++;
			float score = getOrientedChamferScore(templates[it->first.folderIndex][it->first.templateIndex], unit, averageEdges);
			if (score > bestChamferScore) {
				bestChamferScore = score;
				bestTemplateIndex = it->first;
			}
		}
	}

	if (moreTimesThanThetaV > 0 /*&& bestChamferScore >= 0.05*/) {
		//std::printf("Total candidates %4d - more than 3: %2d, best score: %4.5f\n", candidatesCount.size(), moreTimesThanThetaV, bestChamferScore);
		return Candidate(
			x * sceneScaleRatio,
			y * sceneScaleRatio,
			slidingWindowSize * sceneScaleRatio,
			bestTemplateIndex,
			bestChamferScore);
	}
	return Candidate();
}

void nonMaximaSupression(std::vector<Candidate> &candidates) {
	for (int i = 0; i < candidates.size(); i++)
	{
		bool startFromBeginning = false;
		if (!candidates[i].active) { continue; }
		for (int j = i+1; j < candidates.size(); j++)
		{
			if (!candidates[j].active) { continue; }

			if (candidates[i].percentageOverlap(candidates[j]) >= NMSMinOverlap)
			{
				if (candidates[i].chamferScore > candidates[j].chamferScore)
				{
					candidates[j].active = false;
				}
				else
				{
					candidates[i].active = false;
					i = j;
					startFromBeginning = true;
				}
			}
			else if (candidates[i].rect.x + candidates[i].rect.width <= candidates[j].rect.x)
			{
				break;
			}
		}
		if (startFromBeginning)
		{
			i = -1;
		}
	}
}