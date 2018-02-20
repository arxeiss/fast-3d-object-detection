#pragma once

#include <vector>

#include "constantsAndTypes.h"
#include "HashSettings.h"
#include "Triplet.h"

void matchInImage(cv::Mat &testImg_8u, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges);

void matchInImageWithSlidingWindow(cv::Mat &scene_8u, cv::Mat &previewScene, FolderTemplateList &templates, HashSettings &hashSettings, std::vector<Triplet> &triplets, TemplateHashTable &hashTable, float averageEdges, float sceneScaleRatio);
