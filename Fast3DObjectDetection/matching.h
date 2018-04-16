#pragma once

#include <vector>

#include "constantsAndTypes.h"
#include "HashSettings.h"
#include "Triplet.h"
#include "Candidate.h"
#include "GroundTruth.h"
#include "F1Score.h"

F1Score matchInImage(int testIndex, cv::Mat &testImg_8u, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, int minEdges, std::vector<GroundTruth> &groundTruth, bool disableVisualisation = false);

void matchInImageWithSlidingWindow(cv::Mat &scene_8u, std::vector<Candidate> &candidates, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, int minEdges, float sceneScaleRatio);

Candidate computeMatchInSlidingWindow(cv::Mat &scene_8u, cv::Mat &edges_8u, int x, int y, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, int minEdges, float sceneScaleRatio);

int solveBinarySlacification(Candidate &candidate, std::vector<GroundTruth> &grounTruth, F1Score &f1score);

void nonMaximaSupression(std::vector<Candidate> &candidates);
