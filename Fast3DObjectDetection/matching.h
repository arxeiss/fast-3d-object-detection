#pragma once

#include <vector>

#include "constantsAndTypes.h"
#include "HashSettings.h"
#include "Triplet.h"

void matchInImage(cv::Mat testImg_8u, HashSettings &hashSettings, std::vector<Triplet> triplets, TemplateHashTable hashTable);
