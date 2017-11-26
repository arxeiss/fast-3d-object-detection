#pragma once

#include <stddef.h>

#include "DetectionUnit.h"
#include "TemplateChamferScore.h"

float getOrientedChamferScore(DetectionUnit &srcTemplate, DetectionUnit &comparingImage, float averageEdges, float lambda, float thetaD, float thetaPhi, int* matchEdges = NULL);

int compareDescChamferScore(TemplateChamferScore &a, TemplateChamferScore &b);
