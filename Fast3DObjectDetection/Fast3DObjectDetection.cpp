
#include <stdio.h>
#include <tchar.h>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include <iomanip>

#include <omp.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "utils.h"

#include "DetectionUnit.h"
#include "TemplateChamferScore.h"
#include "Triplet.h"
#include "TemplateIndex.h"
#include "TripletValues.h"
#include "QuantizedTripletValues.h"
#include "HashSettings.h"
#include "TimeMeasuring.h"
#include "F1Score.h"

#include "constantsAndTypes.h"

#include "distanceAndOrientation.h"
#include "loading.h"
#include "edgeProcessing.h"
#include "matching.h"
#include "visualize.h"

/*
NOTES:
 - nejlepsi vysledky dist binu = 1.369293, 3.279279, 6.558594
 - prumerny pocet hran je pocitany vuci templatum stejneho objektu - pocitat vuci vsem templatum?
 - prumerny pocet hran pro detekci spocitat uz z vyfiltrovanych sablon, nebo z puvodnich pred odebrani hran?
 - Chamfer score pociat z odfiltrovane sablony proti sliding window - mene hran, rychlejsi beh
*/


void prepareAndSaveData() {
	srand(time(0));
	TimeMeasuring elapsedTime(true);

	FolderTemplateList templates;
	int templatesLoaded = loadAllTemplates(templates);
	elapsedTime.insertBreakpoint("tplLoaded");
	std::printf("%d templates loaded in: %d [ms]\n", templatesLoaded, elapsedTime.getTimeFromBeginning());

	std::printf("\n\n\n\n");
	int minOtherQEdges = 3000;
	for (int f = 0; f < templates.size(); f++)
	{
		for (int t = 0; t < templates[f].size(); t++)
		{
			int q1 = 0, q2 = 0, q3 = 0, q4 = 0;
			for (int x = 0; x < templates[f][t].edges_8u.cols; x++)
			{
				for (int y = 0; y < templates[f][t].edges_8u.rows; y++)
				{
					if (templates[f][t].edges_8u.at<uchar>(y, x) == 0) {
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
			int minQEdges = 8;
			if (q1 < minQEdges || q2 < minQEdges || q3 < minQEdges || q4 < minQEdges) {
				int lessQEdges = 0;
				if (q1 < minQEdges) { lessQEdges++; }
				if (q2 < minQEdges) { lessQEdges++; }
				if (q3 < minQEdges) { lessQEdges++; }
				if (q4 < minQEdges) { lessQEdges++; }
				if (lessQEdges > 1)
				{
					std::printf("TPL %d / %4d Zeros: %d   =   (Q1-4 = %2d - %2d - %2d - %2d)\n", f, t, lessQEdges, q1, q2, q3, q4);
				}
			}
		}
	}

	float averageEdges = countAverageEdgesAcrossTemplates(templates);
	std::printf("Average edges: %f - Min edges: %d\n", averageEdges);
	//showChamferScore(templates[0][0], templates[0][1], averageEdges);
	std::vector<Triplet> triplets = generateTriplets();
	elapsedTime.insertBreakpoint("genTriplets");
	std::printf("Triplets generated in: %d [us] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("tplLoaded", true), elapsedTime.getTimeFromBeginning());
	//visualizeTriplets(triplets, pointsEdgeOffset, pointsDistance, 48);

	TemplateHashTable hashTable;
	HashSettings hashSettings = fillHashTable(hashTable, templates, templatesLoaded, triplets);
	elapsedTime.insertBreakpoint("hashTable");
	std::printf("Method fillHashTable in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("genTriplets"), elapsedTime.getTimeFromBeginning());
	std::printf("hash table size: %d\n", hashTable.size());
	int maxSize = 0;
	for (auto it = hashTable.begin(); it != hashTable.end(); ++it) {
		if (it->second.size() > maxSize)
		{
			maxSize = it->second.size();
		}
	}
	std::printf("Max size: %d\n\n", maxSize);

	// paralel je jiz uvnitr funkce filterTemplateEdges
	// #pragma omp parallel for
	for (int f = 0; f < templates.size(); f++)
	{
		std::printf("Start %d - ", f);
		TimeMeasuring elapsedTime;
		elapsedTime.startMeasuring();
		//showResized("before", templates[f][0].edges_8u, 3);
		filterTemplateEdges(templates[f], averageEdges);
		//showResized("after", templates[f][0].edges_8u, 3);
		//cv::waitKey();
		std::printf("Folder %d of templates filtered in: %d [ms]\n", f, elapsedTime.getTimeFromBeginning());
	}
	std::printf("Edges filtered in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("hashTable"), elapsedTime.getTimeFromBeginning());
	elapsedTime.insertBreakpoint("filterEdges");
	
	savePreparedData("preparedData.bin", templates, triplets);

	elapsedTime.insertBreakpoint("fileSaving");
	std::printf("File saved in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("filterEdges"), elapsedTime.getTimeFromBeginning());

	/*FolderTemplateList templates;
	std::vector<Triplet> triplets;
	TemplateHashTable hashTable;
	HashSettings hashSettingsLoad = loadPreparedData("preparedData.bin", templates, triplets, hashTable);
	elapsedTime.insertBreakpoint("fileLoading");
	std::printf("File loaded in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("fileSaving"), elapsedTime.getTimeFromBeginning());
	savePreparedData("preparedData2.bin", templates, triplets);*/

	std::printf("Total time: %d [ms]\n", elapsedTime.getTimeFromBeginning());
}

void runMatching(bool disableVisualisation = false) {

	TimeMeasuring elapsedTime(true);

	std::printf("Start loading\n");

	FolderTemplateList templates;
	std::vector<Triplet> triplets;
	TemplateHashTable hashTable;
	HashSettings hashSettings;
	int minEdges = 0;

	bool success = loadPreparedData("preparedData.bin", templates, triplets, hashTable, hashSettings, minEdges);
	float averageEdges = countAverageEdgesAcrossTemplates(templates);
	
	std::printf("Average edges: %f - Min edges: %d\n", averageEdges, minEdges);
 	std::printf("hash table size: %d / buckets: %d\n", hashTable.size(), hashTable.bucket_count());
	std::printf("Data loaded - %s\n", (success ? "sucessfully" : "with error"));

	elapsedTime.insertBreakpoint("fileLoaded");
	std::printf("File loaded in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBeginning(), elapsedTime.getTimeFromBeginning());

	std::printf("Total time: %d [ms]\n", elapsedTime.getTimeFromBeginning());
	elapsedTime.insertBreakpoint("full-matching");

	F1Score total;
	for (int i = 1; i <= 60; i++)
	{
		std::printf("\n#####################\n# Scene %d:\n", i);
		std::vector<GroundTruth> groundTruth;
		loadGroundTruthData(groundTruth, i);
		total += matchInImage(i, loadTestImage_8u(i), templates, hashSettings, triplets, hashTable, averageEdges, minEdges, groundTruth, disableVisualisation);
	}
	std::printf("\n\n#####################\n\nTotal F1 %2.5f (Precision %1.4f / Recal: %1.4f)\nTP: %2d, FP: %2d, FN: %2d\n",
		total.getF1Score(true), total.getPrecision(), total.getRecall(),
		total.truePositive, total.falsePositive, total.falseNegative);
	std::printf("All scenes processed in %3.2f [s]", (float)elapsedTime.getTimeFromBreakpoint("full-matching") / 1000.0f);
}

int main()
{
	std::cout << "Insert index of action to run:\n";
	std::cout << "\t1. Prepare data\n";
	std::cout << "\t2. Run matching\n";
	std::cout << "\t3. Run matching without visualization\n";
	std::cout << "\t4. Start from scratch (prepare and run matching)\n";

	int algo;
	std::cin >> algo;
	std::cin.clear();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	switch (algo)
	{
	case 1:
		prepareAndSaveData();
		break;
	case 2:
		runMatching();
		break;
	case 3:
		runMatching(true);
		break;
	case 4:
		prepareAndSaveData();
		runMatching(true);
		break;
	default:
		std::cout << "Invalid action";
	}
	
	cv::destroyAllWindows();
	std::getc(stdin);
	return 0;
	
	//return testDetectedEdgesAndDistanceTransform();

	//return filterTemplateEdges();
	//return selectingByStabilityToViewpoint();
	//return selectingByEdgeOrientations();

    return 0;
}
