
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
#include "visualize.h"

#include "DetectionUnit.h"
#include "TemplateChamferScore.h"
#include "Triplet.h"
#include "TemplateIndex.h"
#include "TripletValues.h"
#include "QuantizedTripletValues.h"
#include "HashSettings.h"
#include "TimeMeasuring.h"

#include "constantsAndTypes.h"

#include "distanceAndOrientation.h"
#include "loading.h"
#include "edgeProcessing.h"


//  --------------- Preprocessing -------------------
// Remove 2 highest bins


int main()
{
	srand(time(0));
	TimeMeasuring elapsedTime(true);

	FolderTemplateList templates;
	int templatesLoaded = loadAllTemplates(templates);
	elapsedTime.insertBreakpoint("tplLoaded");
	std::printf("Templates loaded in: %d [ms]\n", elapsedTime.getTimeFromBeginning());

	std::vector<Triplet> triplets = generateTriplets(tripletsAmount, pointsInRowCol, pointsEdgeOffset, pointsDistance);
	elapsedTime.insertBreakpoint("genTriplets");
	std::printf("Triplets generated in: %d [us] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("tplLoaded",true), elapsedTime.getTimeFromBeginning()); 	
	//visualizeTriplets(triplets, pointsEdgeOffset, pointsDistance, 48);
	
	TemplateHashTable hashTable;
	HashSettings hashSettings = fillHashTable(hashTable, templates, templatesLoaded, triplets, distanceBins, orientationBins);
	//std::getc(stdin);
	elapsedTime.insertBreakpoint("hashTable");
	std::printf("Hash table filled in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("genTriplets"), elapsedTime.getTimeFromBeginning());
	std::printf("hash table size: %d\n", hashTable.size());
	int maxSize = 0;
	for (auto it = hashTable.begin(); it != hashTable.end(); ++it) {
		if (it->second.size() > maxSize)
		{
			maxSize = it->second.size();
		}
		/*if (it->second.size() > 100)
		{
			QuantizedTripletValues qtz = it->first;
			std::printf("%d = TrpI: %d _ D: %d, %d, %d _ Phi %d, %d, %d\n", it->second.size(),
				qtz.tripletIndex, qtz.d1, qtz.d2, qtz.d3, qtz.phi1, qtz.phi2, qtz.phi3);
			/*for (int i = 0; i < 100; i++)
			{
				visualizeTripletOnEdges(templates[it->second[i].folderIndex][it->second[i].templateIndex], triplets[qtz.tripletIndex], NULL, 300);
			}
			cv::waitKey();*/
		//}
	}
	std::printf("Max size: %d\n\n", maxSize);
	
	//maxSize++;
	//int *sizeCnt = new int[maxSize];
	//for (int f = 0; f < maxSize; f++)
	//{
	//	sizeCnt[f] = 0;
	//}
	//for (auto it = hashTable.begin(); it != hashTable.end(); ++it) {
	//	sizeCnt[it->second.size()]++;
	//}
	//for (int f = 0; f < maxSize; f++)
	//{
	//	if (sizeCnt[f] > 0)
	//	{
	//		std::printf("%d templates under %d keys\n", f, sizeCnt[f]);
	//		//std::getc(stdin);
	//	}
	//}	
	//delete sizeCnt;

	// paralel je jiz uvnitr funkce filterTemplateEdges
	// #pragma omp parallel for
	for (int f = 0; f < templates.size(); f++)
	{
		std::printf("Start %d - ", f);
		TimeMeasuring elapsedTime;
		elapsedTime.startMeasuring();
		//showResized("before", templates[f][0].edges_8u, 3);
		filterTemplateEdges(templates[f], kTpl, lambda, thetaD, thetaPhi, tau, removePixelRatio);
		//showResized("after", templates[f][0].edges_8u, 3);
		//cv::waitKey();
		std::printf("Folder %d of templates filtered in: %d [ms]\n", f, elapsedTime.getTimeFromBeginning());
	}
	elapsedTime.insertBreakpoint("filterEdges");
	std::printf("Edges filtered in: %d [ms] (total time: %d [ms])\n", elapsedTime.getTimeFromBreakpoint("hashTable"), elapsedTime.getTimeFromBeginning());

	std::printf("Total time: %d [ms]\n", elapsedTime.getTimeFromBeginning());
	
	std::getc(stdin);
	return 0;
	
	//return testDetectedEdgesAndDistanceTransform();

	//return filterTemplateEdges();
	//return selectingByStabilityToViewpoint();
	//return selectingByEdgeOrientations();

    return 0;
}
