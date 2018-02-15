#pragma once
#include <vector>

#include "constantsAndTypes.h"
#include "DetectionUnit.h"
#include "Triplet.h"
#include "TripletValues.h"
#include "HashSettings.h"

void prepareDetectionUnit(DetectionUnit &dt, bool renewEdges = false, bool renewDistTransform = false, bool recountEdges = false);

std::vector<Triplet> generateTriplets(const int amount, const int inColRow, const int edgeOffset, const int pointsDistance);

int loadAllTemplates(FolderTemplateList &templates);

DetectionUnit getDetectionUnitByROI(cv::Mat img_8u, int x, int y, int roiSize);

cv::Mat loadTestImage_8u(int imageIndex);

TripletValues getTripletValues(int tripletIndex, Triplet &triplet, DetectionUnit &unit);
TripletValues getTripletValues(int tripletIndex, Triplet &triplet, DetectionUnit &unit, TemplateIndex templateIndex);

void countTripletsValues(std::vector<TripletValues> &tripletsValues, FolderTemplateList &templates, std::vector<Triplet> &triplets,
	int templatesLoaded, std::vector<float> &dBinsRange, float *minD = NULL, float *maxD = NULL, float *minPhi = NULL, float *maxPhi = NULL);

HashSettings fillHashTable(TemplateHashTable &hashTable, FolderTemplateList &templates, int templatesLoaded, std::vector<Triplet> &triplets, int dBins, int phiBins);

QuantizedTripletValues getTableHashKey(HashSettings &hashSettings, DetectionUnit &unit, Triplet &triplet, int tripletIndex);

void savePreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets);

bool loadPreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, HashSettings &hashSettings);