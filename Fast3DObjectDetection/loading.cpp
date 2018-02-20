#include "loading.h"

#include <fstream>

#include "distanceAndOrientation.h"
#include "TimeMeasuring.h"

#include "visualize.h"


// detect edges, distance transform, count edges
void prepareDetectionUnit(DetectionUnit &dt, bool renewEdges, bool renewDistTransform, bool recountEdges) {
	if (dt.edges_8u.empty() || renewEdges)
	{
		dt.edges_8u = getDetectedEdges_8u(dt.img_8u);
		recountEdges = true;
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
	if (dt.edgesCount > 0) {
		if (dt.distanceTransform_32f.empty() || renewDistTransform)
		{
			dt.distanceTransform_32f = getDistanceTransformFromEdges_32f(dt.edges_8u);
		}
	}
	else {
		dt.distanceTransform_32f = cv::Mat(dt.img_8u.rows, dt.img_8u.cols, CV_32F);
		dt.distanceTransform_32f = 0.0f;
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

cv::Mat loadTestImage_8u(int imageIndex) {
	if (imageIndex < 1 || imageIndex > 60) {
		return cv::Mat();
	}
	std::string imageIndexStr = (imageIndex < 10 ? "0" : "") + std::to_string(imageIndex);
	return cv::imread("images/CMP-8objs/test/test_" + imageIndexStr + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
}

DetectionUnit getDetectionUnitByROI(cv::Mat img_8u, int x, int y, int roiSize) {
	DetectionUnit unit{};
	unit.img_8u = img_8u(cv::Rect(x, y, roiSize, roiSize));
	prepareDetectionUnit(unit);
	return unit;
}

TripletValues getTripletValues(int tripletIndex, Triplet &triplet, DetectionUnit &unit) {
	return getTripletValues(tripletIndex, triplet, unit, TemplateIndex(0, 0));
}

TripletValues getTripletValues(int tripletIndex, Triplet &triplet, DetectionUnit &unit, TemplateIndex templateIndex) {
	TripletValues trpVal(tripletIndex, templateIndex);
	getEdgeDistAndOri(unit, triplet.p1.x, triplet.p1.y, trpVal.d1, trpVal.phi1, true);
	getEdgeDistAndOri(unit, triplet.p2.x, triplet.p2.y, trpVal.d2, trpVal.phi2, true);
	getEdgeDistAndOri(unit, triplet.p3.x, triplet.p3.y, trpVal.d3, trpVal.phi3, true);

	return trpVal;
}

void countTripletsValues(std::vector<TripletValues> &tripletsValues, FolderTemplateList &templates, std::vector<Triplet> &triplets,
	int templatesLoaded, std::vector<float> &dBinsRange, float *minD, float *maxD, float *minPhi, float *maxPhi)
{
	float minDTmp = FLT_MAX, maxDTmp = FLT_MIN, minPhiTmp = FLT_MAX, maxPhiTmp = FLT_MIN;
	tripletsValues = std::vector<TripletValues>(templatesLoaded * triplets.size());
	std::vector<float> distances = std::vector<float>(tripletsValues.size() * 3);
	std::printf("TripletsValues - %d, distances - %d\n", tripletsValues.size(), distances.size());

	std::map<float, int> dstTest;

	int templateTripletValI = 0;
	int distancesI = 0;
	for (int f = 0; f < templates.size(); f++)
	{
		TemplateList *listTmpPtr = &templates[f];
		for (int tpl = 0; tpl < listTmpPtr->size(); tpl++)
		{
			for (int trp = 0; trp < triplets.size(); trp++)
			{
				TripletValues trpVal = getTripletValues(trp, triplets[trp], (*listTmpPtr)[tpl], TemplateIndex(f, tpl));				
				tripletsValues[templateTripletValI] = trpVal;
				templateTripletValI++;

				distances[distancesI++] = trpVal.d1;
				distances[distancesI++] = trpVal.d2;
				distances[distancesI++] = trpVal.d3;

				dstTest[trpVal.d1]++;
				dstTest[trpVal.d2]++;
				dstTest[trpVal.d3]++;

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

	for (int i = 20; i < 30; i++)
	{
		std::printf("%d - %1.1f - %f\n", i, i / 100.0f, distances[(int)(distances.size() * (i / 100.0f))]);
	}
	for (int i = 45; i < 55; i++)
	{
		std::printf("%d - %1.1f - %f\n", i, i / 100.0f, distances[(int)(distances.size() * (i / 100.0f))]);
	}
	for (int i = 70; i < 80; i++)
	{
		std::printf("%d - %1.1f - %f\n", i, i / 100.0f, distances[(int)(distances.size() * (i / 100.0f))]);
	}

	dBinsRange.clear();
	dBinsRange.push_back(distances[(int)(distances.size() * 0.25)]);
	dBinsRange.push_back(distances[(int)(distances.size() * 0.50)]);
	dBinsRange.push_back(distances[(int)(distances.size() * 0.75)]);

	if (minD != NULL) { *minD = minDTmp; }
	if (maxD != NULL) { *maxD = maxDTmp; }
	if (minPhi != NULL) { *minPhi = minPhiTmp; }
	if (maxPhi != NULL) { *maxPhi = maxPhiTmp; }

	std::printf("Unique dsts:\n");
	int i = 1;
	for (std::map<float, int>::iterator it = dstTest.begin(); it != dstTest.end(); ++it, i++) {
		std::printf("%6.3f - %6dx\t", it->first, it->second);
		if (i % 3 == 0)
		{
			std::printf("\n");
		}
	}

}

HashSettings fillHashTable(TemplateHashTable &hashTable, FolderTemplateList &templates, int templatesLoaded, std::vector<Triplet> &triplets, int dBins, int phiBins) {
	std::vector<TripletValues> tripletsValues;
	float minD, maxD, minPhi, maxPhi;
	TimeMeasuring tm(true);
	std::vector<float> dBinsRange;
	countTripletsValues(tripletsValues, templates, triplets, templatesLoaded, dBinsRange, &minD, &maxD, &minPhi, &maxPhi);
	std::printf("Count triplet vals in %d[ms]\n", tm.getTimeFromBeginning());
	HashSettings hashSettings(minD, maxD, minPhi, maxPhi, dBins, phiBins);
	hashSettings.dBinsRange = dBinsRange;

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

QuantizedTripletValues getTableHashKey(HashSettings &hashSettings, DetectionUnit &unit, Triplet &triplet, int tripletIndex) {
	TripletValues tripletValues = getTripletValues(tripletIndex, triplet, unit);
	QuantizedTripletValues hashKey(tripletValues.tripletIndex,
		hashSettings.getDistanceBin(tripletValues.d1),
		hashSettings.getDistanceBin(tripletValues.d2),
		hashSettings.getDistanceBin(tripletValues.d3),
		hashSettings.getOrientationBin(tripletValues.phi1),
		hashSettings.getOrientationBin(tripletValues.phi2),
		hashSettings.getOrientationBin(tripletValues.phi3));

	return hashKey;
}

void savePreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets, float averageEdges) {
	std::ofstream ofs(fileName, std::ios::binary);
	if (!ofs.is_open()) {
		std::printf("!!! - Error, cannot open a file %s for writing\n", fileName);
	}

	int tripletsSize = triplets.size();
	ofs.write((const char*)(&tripletsSize), sizeof(int));
	for (int t = 0; t < tripletsSize; t++)
	{
		ofs.write((const char*)(&triplets[t].p1.x), sizeof(int));
		ofs.write((const char*)(&triplets[t].p1.y), sizeof(int));
		ofs.write((const char*)(&triplets[t].p2.x), sizeof(int));
		ofs.write((const char*)(&triplets[t].p2.y), sizeof(int));
		ofs.write((const char*)(&triplets[t].p3.x), sizeof(int));
		ofs.write((const char*)(&triplets[t].p3.y), sizeof(int));
	}

	int folders = templates.size(),
		templatesPerFolder = templates[0].size();
	ofs.write((const char*)(&folders), sizeof(int));
	ofs.write((const char*)(&templatesPerFolder), sizeof(int));

	for (int f = 0; f < folders; f++)
	{
		for (int t = 0; t < templatesPerFolder; t++)
		{
			cv::Mat *mat = &(templates[f][t].img_8u);
			int type = mat->type();
			ofs.write((const char*)(&mat->rows), sizeof(int));
			ofs.write((const char*)(&mat->cols), sizeof(int));
			ofs.write((const char*)(&type), sizeof(int));
			ofs.write((const char*)(mat->data), mat->elemSize() * mat->total());
		}
	}

	for (int f = 0; f < folders; f++)
	{
		for (int t = 0; t < templatesPerFolder; t++)
		{
			cv::Mat *mat = &(templates[f][t].edges_8u);
			int type = mat->type();
			ofs.write((const char*)(&mat->rows), sizeof(int));
			ofs.write((const char*)(&mat->cols), sizeof(int));
			ofs.write((const char*)(&type), sizeof(int));
			ofs.write((const char*)(mat->data), mat->elemSize() * mat->total());
		}
	}

	ofs.write((const char*)(&averageEdges), sizeof(float));

	ofs.close();
}
// https://github.com/takmin/BinaryCvMat/blob/master/BinaryCvMat.cpp
// http://pythonopencv.com/step-by-step-install-opencv-3-3-with-visual-studio-2015-on-windows-10-x64-2017-diy/
// http://pythonopencv.com/easy-fast-pre-compiled-opencv-libraries-and-headers-for-3-2-with-visual-studio-2015-x64-windows-10-support/
bool loadPreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, HashSettings &hashSettings, float &averageEdges) {
	std::ifstream ifsData(fileName, std::ios::binary);

	if (!ifsData.is_open()) {
		std::printf("!!! - Error, cannot open a file %s for reading\n", fileName);
		return false;
	}

	int tripletsSize;
	ifsData.read((char*)(&tripletsSize), sizeof(int));
	triplets.reserve(tripletsSize);
	for (int i = 0; i < tripletsSize; i++)
	{
		cv::Point p1, p2, p3;
		ifsData.read((char*)(&p1.x), sizeof(int));
		ifsData.read((char*)(&p1.y), sizeof(int));
		ifsData.read((char*)(&p2.x), sizeof(int));
		ifsData.read((char*)(&p2.y), sizeof(int));
		ifsData.read((char*)(&p3.x), sizeof(int));
		ifsData.read((char*)(&p3.y), sizeof(int));
		triplets.push_back(Triplet(p1, p2, p3));
	}

	int folders, templatesPerFolder;
	ifsData.read((char*)(&folders), sizeof(int));
	ifsData.read((char*)(&templatesPerFolder), sizeof(int));

	templates.resize(folders);
	for (int f = 0; f < folders; f++)
	{
		templates[f].reserve(templatesPerFolder);
		for (int t = 0; t < templatesPerFolder; t++)
		{
			int rows, cols, type;
			ifsData.read((char*)(&rows), sizeof(int));
			ifsData.read((char*)(&cols), sizeof(int));
			ifsData.read((char*)(&type), sizeof(int));

			DetectionUnit unit{};
			unit.img_8u.create(rows, cols, type);
			ifsData.read((char*)(unit.img_8u.data), unit.img_8u.elemSize() * unit.img_8u.total());

			prepareDetectionUnit(unit, true, true, true);
			templates[f].push_back(unit);
		}
	}
	int templatesLoaded = folders * templatesPerFolder;
	hashSettings = fillHashTable(hashTable, templates, templatesLoaded, triplets, distanceBins, orientationBins);

	for (int f = 0; f < folders; f++)
	{
		for (int t = 0; t < templatesPerFolder; t++)
		{
			int rows, cols, type;
			ifsData.read((char*)(&rows), sizeof(int));
			ifsData.read((char*)(&cols), sizeof(int));
			ifsData.read((char*)(&type), sizeof(int));

			cv::Mat *edges = &(templates[f][t].edges_8u);
			edges->release();
			edges->create(rows, cols, type);
			ifsData.read((char*)(edges->data), edges->elemSize() * edges->total());
		}
	}
	ifsData.read((char*)(&averageEdges), sizeof(float));

	ifsData.close();

	return true;
}