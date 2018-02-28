#pragma once

#include <vector>

#include "constantsAndTypes.h"
#include "HashSettings.h"
#include "Triplet.h"
#include "Candidate.h"

void matchInImage(cv::Mat &testImg_8u, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges);

void matchInImageWithSlidingWindow(cv::Mat &scene_8u, std::vector<Candidate> &candidates, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, float sceneScaleRatio);

Candidate computeMatchInSlidingWindow(cv::Mat &scene_8u, int x, int y, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, float sceneScaleRatio);

void nonMaximaSupression(std::vector<Candidate> &candidates);
