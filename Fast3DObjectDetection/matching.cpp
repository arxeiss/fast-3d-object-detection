#include "matching.h"

#include "visualize.h"
#include "utils.h"
#include "loading.h"
#include "TimeMeasuring.h"
#include "chamferScore.h"

#include <fstream>

F1Score matchInImage(int testIndex, cv::Mat &testImg_8u, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, int minEdges, std::vector<GroundTruth> &groundTruth, bool disableVisualisation) {
	
	cv::Mat toDraw;

	std::fstream timeData("matchInImage-time.log", std::fstream::out | std::fstream::app);
	if (timeData.is_open()) {
		if (testIndex == 1)
		{
			timeData << "i,s9,s8,s7,s6,s5,s4,s3,s2,s1,s0,NMS[us],windows,winAfterNms|f1score,tp,fp,fn\n";
		}
		timeData << testIndex << ",";
	} else {
		std::printf("!!! - Error, cannot open a file for writing !!!\n");
	}

	TimeMeasuring tm;
	tm.startMeasuring();

	std::vector<Candidate> candidates;

	minEdges = floor((float)minEdges * minEdgesRatio);

	
	cv::Mat sizedScene;
	testImg_8u.copyTo(sizedScene);
	blurImage(sizedScene);

	//for (int i = scalePyramidSteps - 1; i >= 0; i--)
	for (int i = 0; i < scalePyramidSteps; i++)
	{
		
		TimeMeasuring tm(true);
		float scaleRatio = fastPow(scalePyramidResizeRatio, i);
		
		printf("Scale pyramid - step %d @ %dx%d", i, sizedScene.cols, sizedScene.rows);
		matchInImageWithSlidingWindow(sizedScene, candidates, templates, hashSettings, triplets, hashTable, averageEdges, minEdges, scaleRatio);
		if (i < scalePyramidSteps - 1)
		{
			cv::resize(sizedScene, sizedScene, cv::Size(round(sizedScene.cols / scalePyramidResizeRatio), round(sizedScene.rows / scalePyramidResizeRatio)));
		}
		//cv::Mat canny = getDetectedEdges_8u(sizedScene);
		//cv::imshow("Canny", canny);
		//cv::waitKey();
		long long int ms = tm.getTimeFromBeginning();
		printf(" => in %d [ms]\n", ms);
		/*testImg_8u.copyTo(toDraw);
		for (int i = 0; i < candidates.size(); i++)
		{
			drawSlidingWindowToImage(toDraw, candidates[i].rect.width, candidates[i].rect.x, candidates[i].rect.y, candidates[i].chamferScore * 4);
		}
		cv::imshow("Possible candidates", toDraw);
		cv::waitKey();*/
		if (timeData.is_open()) {
			timeData << ms << ",";
		}
	}
	std::printf("\nTotal matching time: %d [ms]\n", tm.getTimeFromBeginning());
	if (!disableVisualisation)
	{
		testImg_8u.copyTo(toDraw);
		for (int i = 0; i < candidates.size(); i++)
		{
			drawSlidingWindowToImage(toDraw, candidates[i].rect.width, candidates[i].rect.x, candidates[i].rect.y, candidates[i].chamferScore * 4);
		}
		cv::imshow("Possible candidates", toDraw);
	}
	std::printf("Total windows: %d\n", candidates.size());	

	tm.insertBreakpoint("nms");
	std::sort(candidates.begin(), candidates.end());
	nonMaximaSupression(candidates);
	long long int us = tm.getTimeFromBreakpoint("nms", true);
	if (timeData.is_open()) {
		timeData << us << ",";
	}
	std::printf("NMS time: %d [us]\n", us);

	F1Score imageScore;
	std::sort(groundTruth.begin(), groundTruth.end());
	
	cv::Mat NMS;
	if (!disableVisualisation)
	{
		testImg_8u.copyTo(NMS);
	}
	int windows = 0;
	for (int i = 0; i < candidates.size(); i++)
	{
		if (candidates[i].active)
		{
			windows++;

			int groundTruthI = solveBinarySlacification(candidates[i], groundTruth, imageScore);

			if (!disableVisualisation)
			{
				if (groundTruthI >= 0) {
					drawWindowToImage(NMS, groundTruth[groundTruthI].rect, cv::Scalar(0, 255, 0));
				}

				std::stringstream ss;
				ss << "F: " << candidates[i].tplIndex.folderIndex << " T: " << candidates[i].tplIndex.templateIndex;
				drawSlidingWindowToImage(NMS, candidates[i].rect.width, candidates[i].rect.x, candidates[i].rect.y, candidates[i].chamferScore * 4, ss.str());
				TemplateIndex &tplIndex = candidates[i].tplIndex;
				getEdgesAndDrawFullSizeToSource(NMS,
					templates[tplIndex.folderIndex][tplIndex.templateIndex].img_8u,
					candidates[i].rect.x,
					candidates[i].rect.y,
					(float)candidates[i].rect.width / (float)slidingWindowSize,1, cv::Vec3b(0,0,255));
			}
		}
	}
	
	for (int i = 0; i < groundTruth.size(); i++)
	{
		if (!groundTruth[i].active)
		{
			continue;
		}
		imageScore.falseNegative++;
		if (!disableVisualisation)
		{
			drawWindowToImage(NMS, groundTruth[i].rect, cv::Scalar(0, 255, 0));
		}
	}

	std::printf("Total windows after NMS: %d\n", windows);
	if (timeData.is_open()) {
		timeData << candidates.size() << "," << windows << "|" << imageScore.getF1Score(true) << ","
			<< imageScore.truePositive << "," << imageScore.falsePositive << "," << imageScore.falseNegative << "\n";
		timeData.close();
	}

	std::printf("# F1 %2.3f (Precision %1.4f / Recal: %1.4f) - TP: %2d, FP: %2d, FN: %2d\n",
		imageScore.getF1Score(true), imageScore.getPrecision(), imageScore.getRecall(),
		imageScore.truePositive, imageScore.falsePositive, imageScore.falseNegative);
	if (!disableVisualisation) {
		cv::imshow("Detection", NMS);
		cv::waitKey();
	}
	return imageScore;
}

void matchInImageWithSlidingWindow(cv::Mat &scene_8u, std::vector<Candidate> &candidates, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, int minEdges, float sceneScaleRatio) {
	int maxX = scene_8u.cols - slidingWindowSize + 1;
	int maxY = scene_8u.rows - slidingWindowSize + 1;

	cv::Mat edges_8u = getDetectedEdges_8u(scene_8u);

#pragma omp parallel for
	for (int x = 0; x < maxX; x += slidingWindowStep)
	{
		for (int y = 0; y < maxY; y += slidingWindowStep)
		{
			Candidate candidate = computeMatchInSlidingWindow(scene_8u, edges_8u, x, y, templates, hashSettings, triplets, hashTable, averageEdges, minEdges, sceneScaleRatio);
			if (candidate.active)
			{			
				#pragma omp critical
				candidates.push_back(candidate);
			}
		}
	}
}

Candidate computeMatchInSlidingWindow(cv::Mat &scene_8u, cv::Mat &edges_8u, int x, int y, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, int minEdges, float sceneScaleRatio) {
	DetectionUnit unit = getDetectionUnitByROI(scene_8u, edges_8u, x, y, slidingWindowSize);
	if (unit.edgesCount < minEdges) {
		return Candidate();
	}
	int q1 = 0, q2 = 0, q3 = 0, q4 = 0;
	for (int x = 0; x < unit.edges_8u.cols; x++)
	{
		for (int y = 0; y < unit.edges_8u.rows; y++)
		{
			if (unit.edges_8u.at<uchar>(y, x) == 0) {
				if (x < 24) {
					if (y < 24) { q1++; }
					else { q2++; }
				}
				else
				{
					if (y < 24) { q3++; }
					else { q4++; }
				}
			}
		}
	}
	int minQEdges = 10;
	if (q1 < minQEdges || q2 < minQEdges || q3 < minQEdges || q4 < minQEdges) {
		int lessQEdges = 0;
		if (q1 < minQEdges) { lessQEdges++; }
		if (q2 < minQEdges) { lessQEdges++; }
		if (q3 < minQEdges) { lessQEdges++; }
		if (q4 < minQEdges) { lessQEdges++; }
		if (lessQEdges > 1)
		{
			return Candidate();
		}
	}

	std::unordered_map<TemplateIndex, int> candidatesCount;
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

	if (moreTimesThanThetaV > 0 && bestChamferScore > 0.5) {
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

int solveBinarySlacification(Candidate &candidate, std::vector<GroundTruth> &grounTruth, F1Score &f1score) {
	F1Score score;
	for (int i = 0; i < grounTruth.size(); i++)
	{	
		if (candidate.rect.x + candidate.rect.width <= grounTruth[i].rect.x)
		{
			break;
		}
		if (grounTruth[i].intersectOverUnion(candidate) >= GTMinOverlap)
		{
			if (candidate.folderIndex == grounTruth[i].folderIndex)
			{
				f1score.truePositive++;
				/*std::fstream log("tp.csv", std::fstream::app);
				log << candidate.chamferScore << "\n";
				log.close();*/
			}
			else {
				f1score.falseNegative++;
				/*std::fstream log("fn.csv", std::fstream::app);
				log << candidate.chamferScore << "\n";
				log.close();*/
			}
			grounTruth[i].active = false;
			return i;
		}
	}
	f1score.falsePositive++;
	/*std::fstream log("fp.csv", std::fstream::app);
	log << candidate.chamferScore << "\n";
	log.close();*/
	return -1;
}

void nonMaximaSupression(std::vector<Candidate> &candidates) {
	for (int i = 0; i < candidates.size(); i++)
	{
		bool startFromBeginning = false;
		if (!candidates[i].active) { continue; }
		for (int j = i+1; j < candidates.size(); j++)
		{
			if (!candidates[j].active) { continue; }
			if (candidates[i].rect.x + candidates[i].rect.width <= candidates[j].rect.x) { break; }

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
		}
		if (startFromBeginning)
		{
			i = -1;
		}
	}
}