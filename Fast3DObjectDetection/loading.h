#pragma once
#include <vector>

#include "constantsAndTypes.h"
#include "DetectionUnit.h"
#include "Triplet.h"
#include "TripletValues.h"
#include "HashSettings.h"
#include "GroundTruth.h"

void prepareDetectionUnit(DetectionUnit &dt, bool recountEdges = false, bool renewEdges = false, bool renewDistTransform = false);

void blurImage(cv::Mat &img);

std::vector<Triplet> generateTriplets();

void thresholdToValue(cv::Mat &dst_8u, uchar noUnderThreshold);

int loadAllTemplates(FolderTemplateList &templates);

DetectionUnit getDetectionUnitByROI(cv::Mat &img_8u, cv::Mat &edges_8u, int x, int y, int roiSize);

cv::Mat loadTestImage_8u(int imageIndex);
void loadGroundTruthData(std::vector<GroundTruth> &groundTruth, int imageIndex);

TripletValues getTripletValues(int tripletIndex, Triplet &triplet, DetectionUnit &unit);
TripletValues getTripletValues(int tripletIndex, Triplet &triplet, DetectionUnit &unit, TemplateIndex &templateIndex);

void countTripletsValues(std::vector<TripletValues> &tripletsValues, FolderTemplateList &templates, std::vector<Triplet> &triplets,
	int templatesLoaded, std::vector<float> &dBinsRange, float *minD = NULL, float *maxD = NULL, float *minPhi = NULL, float *maxPhi = NULL);

HashSettings fillHashTable(TemplateHashTable &hashTable, FolderTemplateList &templates, int templatesLoaded, std::vector<Triplet> &triplets);

QuantizedTripletValues getTableHashKey(HashSettings &hashSettings, DetectionUnit &unit, Triplet &triplet, int tripletIndex);

void savePreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets);

bool loadPreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, HashSettings &hashSettings, int &minEdges);