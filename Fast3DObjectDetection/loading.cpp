#include "loading.h"

#include "distanceAndOrientation.h"
#include "TimeMeasuring.h"

// detect edges, distance transform, count edges
void prepareDetectionUnit(DetectionUnit &dt, bool renewEdges, bool renewDistTransform, bool recountEdges) {
	if (dt.edges_8u.empty() || renewEdges)
	{
		dt.edges_8u = getDetectedEdges_8u(dt.img_8u);
		recountEdges = true;
	}
	if (dt.distanceTransform_32f.empty() || renewDistTransform)
	{
		dt.distanceTransform_32f = getDistanceTransformFromEdges_32f(dt.edges_8u);
	}
	if (dt.edgesCount == 0 || recountEdges) {
		for (int x = 0; x < dt.edges_8u.cols; x++)
		{
			for (int y = 0; y < dt.edges_8u.rows; y++)
			{
				if (dt.edges_8u.at<uchar>(y, x) == 0) {
					dt.edgesCount++;
				}
			}
		}
	}
}

std::vector<Triplet> generateTriplets(const int amount, const int inColRow, const int edgeOffset, const int pointsDistance)
{
	const int tripletPoints = inColRow * inColRow;
	std::vector<Triplet> triplets;
	for (int i = 0; i < amount; i++)
	{
		int p1 = rand() % tripletPoints, p2 = rand() % tripletPoints, p3 = rand() % tripletPoints;
		while (p1 == p2)
		{
			p2 = rand() % tripletPoints;
		}
		while (p1 == p3 || p2 == p3)
		{
			p3 = rand() % tripletPoints;
		}

		Triplet newTriplet(cv::Point((p1 % inColRow) * pointsDistance + edgeOffset, (p1 / inColRow) * pointsDistance + edgeOffset),
			cv::Point((p2 % inColRow) * pointsDistance + edgeOffset, (p2 / inColRow) * pointsDistance + edgeOffset),
			cv::Point((p3 % inColRow) * pointsDistance + edgeOffset, (p3 / inColRow) * pointsDistance + edgeOffset));
		bool collision = false;
		for (int t = 0; t < triplets.size(); t++)
		{
			if (triplets[t] == newTriplet)
			{
				collision = true;
				break;
			}
		}
		if (collision)
		{
			i--;
			continue;
		}
		triplets.push_back(newTriplet);
	}
	return triplets;
}

int loadAllTemplates(FolderTemplateList &templates) {
	std::string folders[] = {
		"images/CMP-8objs/train-opt2/block/",
		"images/CMP-8objs/train-opt2/bridge/",
		"images/CMP-8objs/train-opt2/cup/",
		"images/CMP-8objs/train-opt2/driver/",
		"images/CMP-8objs/train-opt2/eye/",
		"images/CMP-8objs/train-opt2/lid/",
		"images/CMP-8objs/train-opt2/screw/",
		"images/CMP-8objs/train-opt2/whiteblock/"
	};
	int templatesInFolder = 1620;
	templates = FolderTemplateList(sizeof(folders) / sizeof(folders[0]));
	int templatesLoaded = 0;

#pragma omp parallel for
	for (int f = 0; f < templates.size(); f++)
	{
		templates[f] = TemplateList(templatesInFolder);
		TimeMeasuring elapsedTime(true);

		for (int t = 1; t <= templatesInFolder; t++)
		{
			DetectionUnit unit{};
			std::string templateName = std::to_string(t);
			templateName.insert(templateName.begin(), 5 - templateName.size(), '0');
			unit.img_8u = cv::imread(folders[f] + "template_" + templateName + ".png", CV_LOAD_IMAGE_GRAYSCALE);
			prepareDetectionUnit(unit);
			templates[f][t - 1] = unit;
#pragma omp critical
			templatesLoaded++;
		}

		std::printf("Folder \"%s\" loaded in %d [ms]\n", folders[f].c_str(), elapsedTime.getTimeFromBeginning());

	}
	return templatesLoaded;
}

void countTripletsValues(std::vector<TripletValues> &tripletsValues, FolderTemplateList &templates, std::vector<Triplet> &triplets,
	int templatesLoaded, float *minD, float *maxD, float *minPhi, float *maxPhi)
{
	float minDTmp = FLT_MAX, maxDTmp = FLT_MIN, minPhiTmp = FLT_MAX, maxPhiTmp = FLT_MIN;
	tripletsValues = std::vector<TripletValues>(templatesLoaded * triplets.size());
	std::vector<float> distances = std::vector<float>(tripletsValues.size() * 3);

	int templateTripletValI = 0;
	int distancesI = 0;
	for (int f = 0; f < templates.size(); f++)
	{
		TemplateList *listTmpPtr = &templates[f];
		for (int tpl = 0; tpl < listTmpPtr->size(); tpl++)
		{
			for (int trp = 0; trp < triplets.size(); trp++)
			{
				TripletValues trpVal(trp, TemplateIndex(f, tpl));
				getEdgeDistAndOri((*listTmpPtr)[tpl], triplets[trp].p1.x, triplets[trp].p1.y, trpVal.d1, trpVal.phi1, true);
				getEdgeDistAndOri((*listTmpPtr)[tpl], triplets[trp].p2.x, triplets[trp].p2.y, trpVal.d2, trpVal.phi2, true);
				getEdgeDistAndOri((*listTmpPtr)[tpl], triplets[trp].p3.x, triplets[trp].p3.y, trpVal.d3, trpVal.phi3, true);
				tripletsValues[templateTripletValI] = trpVal;
				templateTripletValI++;

				distances[distancesI++] = trpVal.d1;
				distances[distancesI++] = trpVal.d2;
				distances[distancesI++] = trpVal.d3;

				float tripletMinD = trpVal.minDistance(),
					tripletMaxD = trpVal.maxDistance(),
					tripletMinPhi = trpVal.minOrientation(),
					tripletMaxPhi = trpVal.maxOrientation();
				if (tripletMinD < minDTmp) { minDTmp = tripletMinD; }
				if (tripletMaxD > maxDTmp) { maxDTmp = tripletMaxD; }
				if (tripletMinPhi < minPhiTmp) { minPhiTmp = tripletMinPhi; }
				if (tripletMaxPhi > maxPhiTmp) { maxPhiTmp = tripletMaxPhi; }


				/*std::printf("Folder: %d, Template: %d (%d), Triplet %d\n", f, tpl, trp);
				std::printf("d1: %4.2f, d2: %4.2f, d3: %4.2f, phi1: %4.2f, phi2: %4.2f, phi3: %4.2f\n",
				trpVal.d1, trpVal.d2, trpVal.d3, trpVal.phi1, trpVal.phi2, trpVal.phi3);
				std::printf("Min d: %4.2f, phi %4.2f   Max: d: %4.2f, phi: %4.2f\n\n", tripletMinD, tripletMinPhi, tripletMaxD, tripletMaxPhi);*/
				//visualizeTripletOnEdges((*listTmpPtr)[tpl], triplets[trp], &trpVal);
			}
		}
	}
	TimeMeasuring tm(true);
	std::sort(distances.begin(), distances.end());
	std::printf("Sorted in %d[ms] - length: %d\n", tm.getTimeFromBeginning(), distances.size());
	std::printf("quantile 0.25 = %f\nquantile 0.5 = %f\nquantile 0.75 = %f\n\n", distances[(int)(distances.size() * 0.25)],
		distances[(int)(distances.size() * 0.5)], distances[(int)(distances.size() * 0.75)]);

	if (minD != NULL) { *minD = minDTmp; }
	if (maxD != NULL) { *maxD = maxDTmp; }
	if (minPhi != NULL) { *minPhi = minPhiTmp; }
	if (maxPhi != NULL) { *maxPhi = maxPhiTmp; }
}

HashSettings fillHashTable(TemplateHashTable &hashTable, FolderTemplateList &templates, int templatesLoaded, std::vector<Triplet> &triplets, int dBins, int phiBins) {
	std::vector<TripletValues> tripletsValues;
	float minD, maxD, minPhi, maxPhi;
	TimeMeasuring tm(true);
	countTripletsValues(tripletsValues, templates, triplets, templatesLoaded, &minD, &maxD, &minPhi, &maxPhi);
	std::printf("Count triplet vals in %d[ms]\n", tm.getTimeFromBeginning());
	HashSettings hashSettings(minD, maxD, minPhi, maxPhi, dBins, phiBins);

	std::printf("\nTripletsValues: %d\n\n", tripletsValues.size());
	std::printf("Min d: %4.2f, phi %4.2f   Max: d: %4.2f, phi: %4.2f\n\n", minD, minPhi, maxD, maxPhi);

	int const bins = 6;
	int dBinsCnt[bins] = { 0,0,0,0,0,0 };
	int phiBinsCnt[bins] = { 0,0,0,0,0,0 };

	for (int i = 0; i < tripletsValues.size(); i++)
	{
		QuantizedTripletValues hashKey(tripletsValues[i].tripletIndex,
			hashSettings.getDistanceBin(tripletsValues[i].d1),
			hashSettings.getDistanceBin(tripletsValues[i].d2),
			hashSettings.getDistanceBin(tripletsValues[i].d3),
			hashSettings.getOrientationBin(tripletsValues[i].phi1),
			hashSettings.getOrientationBin(tripletsValues[i].phi2),
			hashSettings.getOrientationBin(tripletsValues[i].phi3));
		hashTable[hashKey].push_back(tripletsValues[i].templateIndex);


		dBinsCnt[hashKey.d1]++;
		dBinsCnt[hashKey.d2]++;
		dBinsCnt[hashKey.d3]++;
		phiBinsCnt[hashKey.phi1]++;
		phiBinsCnt[hashKey.phi2]++;
		phiBinsCnt[hashKey.phi3]++;
	}


	std::printf("Distance bins:\n");
	for (int i = 0; i < bins; i++)
	{
		std::printf("\t%d:%8d\n", i, dBinsCnt[i]);
	}
	std::printf("\nOrientation bins:\n");
	for (int i = 0; i < bins; i++)
	{
		std::printf("\t%d:%8d\n", i, phiBinsCnt[i]);
	}
	std::printf("\n");

	return hashSettings;
}
