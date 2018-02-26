#pragma once

#include "constantsAndTypes.h"
#include "DetectionUnit.h"

cv::Mat removeDiscriminatePoints_8u(cv::Mat &src_8u, cv::Mat &edge_8u, float removePixelRatio);

int selectingByEdgeOrientations();

cv::Mat removeNonStablePoints_8u(DetectionUnit &srcTemplate, std::vector<DetectionUnit> &simmilarTemplates, float thetaD, float thetaPhi, float tau);

void filterTemplateEdges(std::vector<DetectionUnit> &templates, float averageEdges);

float countAverageEdgesAcrossTemplates(FolderTemplateList &folderTemplates);