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

void countTripletsValues(std::vector<TripletValues> &tripletsValues, FolderTemplateList &templates, std::vector<Triplet> &triplets,
	int templatesLoaded, std::vector<float> &dBinsRange, float *minD = NULL, float *maxD = NULL, float *minPhi = NULL, float *maxPhi = NULL);

HashSettings fillHashTable(TemplateHashTable &hashTable, FolderTemplateList &templates, int templatesLoaded, std::vector<Triplet> &triplets, int dBins, int phiBins);

void savePreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets);

bool loadPreparedData(std::string fileName, FolderTemplateList &templates, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, HashSettings &hashSettings);