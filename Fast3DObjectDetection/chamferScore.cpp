#include "chamferScore.h"

#include "distanceAndOrientation.h"

float getOrientedChamferScore(DetectionUnit &srcTemplate, DetectionUnit &comparingImage, float averageEdges, float lambda, float thetaD, float thetaPhi, int* matchEdges) {
	int edges = 0;
	for (int x = 0; x < srcTemplate.edges_8u.cols; x++)
	{
		for (int y = 0; y < srcTemplate.edges_8u.rows; y++)
		{
			if (srcTemplate.edges_8u.at<uchar>(y, x) > 0)
			{
				continue;
			}
			float distance = comparingImage.distanceTransform_32f.at<float>(y, x);
			if (distance > thetaD) {
				continue;
			}
			float angleT = getEdgeOrientation(srcTemplate.img_8u, x, y, true);
			float angleI = (distance == 0.0f) ?
				getEdgeOrientation(comparingImage.img_8u, x, y, true) :
				getEdgeOrientationFromDistanceTransform(srcTemplate.distanceTransform_32f, x, y, true);
			if (abs(angleT - angleI) <= thetaPhi)
			{
				edges++;
			}
		}
	}
	float denominator = (lambda * (float)srcTemplate.edgesCount) + ((1 - lambda) * averageEdges);
	if (matchEdges != NULL) {
		*matchEdges = edges;
	}
	return (float)edges / denominator;
}

int compareDescChamferScore(TemplateChamferScore &a, TemplateChamferScore &b)
{
	// Reverse order
	return !(a.chamferScore < b.chamferScore);
}