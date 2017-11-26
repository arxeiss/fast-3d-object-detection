#include "edgeProcessing.h"

#include "constantsAndTypes.h"
#include "utils.h"
#include "distanceAndOrientation.h"
#include "chamferScore.h"
#include "loading.h"

cv::Mat removeDiscriminatePoints_8u(cv::Mat &src_8u, cv::Mat &edge_8u, float removePixelRatio) {
	cv::Mat filtered_8u;
	edge_8u.copyTo(filtered_8u);
	// 15, 45, 75, 105, 135, 165
	// 0-30, 30-60, 60-90, 90-120, 120-150, 150
	int const histogramBins = 6;
	int const removeBins = 2;
	float const ranges[histogramBins - 1] = { M_PI / 6.0f, M_PI / 3.0f, M_PI / 2.0f, 2 * M_PI / 3.0f, 5 * M_PI / 6.0f };

	std::vector<cv::Point2i> histogram[histogramBins];
	int pixelsToCnt = 0;

	int offset = 1;
	for (int x = offset; x < edge_8u.cols - offset; x++)
	{
		for (int y = offset; y < edge_8u.rows - offset; y++)
		{
			uchar pixelVal = edge_8u.at<uchar>(y, x);
			if (pixelVal >= 1)
			{
				continue;
			}

			pixelsToCnt++;
			int histBinIndex = histogramBins - 1;
			float angle = getEdgeOrientation(src_8u, x, y, true);
			for (int i = 0; i < histogramBins - 1; i++)
			{
				if (angle < ranges[i])
				{
					histBinIndex = i;
					break;
				}
			}
			histogram[histBinIndex].push_back(cv::Point2i(x, y));
		}
	}

	/*for (int i = 0; i < histogramBins; i++)
	{
	std::printf("%d: %d\n", i, histogram[i].size());
	}*/

	for (int rm = 0; rm < removeBins; rm++) {
		int maxBinI = 0, maxBinCnt = 0;
		for (int i = 0; i < histogramBins; i++)
		{
			if (maxBinCnt < histogram[i].size())
			{
				maxBinCnt = histogram[i].size();
				maxBinI = i;
			}
		}

		//printf("Remove %d. bin\n", maxBinI);

		int pixelsToRemove = ((float)histogram[maxBinI].size()) * removePixelRatio;
		std::random_shuffle(histogram[maxBinI].begin(), histogram[maxBinI].end());

		for (int i = 0; i < pixelsToRemove; i++)
		{
			cv::Point2i point = histogram[maxBinI].at(i);
			filtered_8u.at<uchar>(point) = 255;
		}
		histogram[maxBinI].clear();
	}

	return filtered_8u;

}
/// TEST func - edges + distance transform;
int selectingByEdgeOrientations() {
	cv::Mat hand_8u = cv::imread("images/hand.png", CV_LOAD_IMAGE_GRAYSCALE);
	printMatType(hand_8u);

	cv::Mat handEdge_8u = getDetectedEdges_8u(hand_8u);
	cv::Mat handEdgeFiltered = removeDiscriminatePoints_8u(hand_8u, handEdge_8u, removePixelRatio);

	//cv::imshow("hand", hand);
	cv::imshow("hand_edge", handEdge_8u);
	cv::imshow("hand_edge_filtered", handEdgeFiltered);

	cv::waitKey(0);
	return 0;
}

cv::Mat removeNonStablePoints_8u(DetectionUnit &srcTemplate, std::vector<DetectionUnit> &simmilarTemplates, float thetaD, float thetaPhi, float tau) {
	int kTpl = simmilarTemplates.size();
	cv::Mat removedPoints_8u;
	srcTemplate.edges_8u.copyTo(removedPoints_8u);
	for (int x = 0; x < srcTemplate.edges_8u.cols; x++)
	{
		for (int y = 0; y < srcTemplate.edges_8u.rows; y++)
		{
			if (srcTemplate.edges_8u.at<uchar>(y, x) > 0)
			{
				continue;
			}
			int edges = 0;
			for (int s = 0; s < kTpl; s++)
			{
				float distance = simmilarTemplates[s].distanceTransform_32f.at<float>(y, x);
				if (distance > thetaD) {
					continue;
				}
				float angleT = getEdgeOrientation(simmilarTemplates[s].img_8u, x, y, true);
				float angleI = (distance == 0.0f) ?
					getEdgeOrientation(simmilarTemplates[s].img_8u, x, y, true) :
					getEdgeOrientationFromDistanceTransform(simmilarTemplates[s].distanceTransform_32f, x, y, true);
				if (abs(angleT - angleI) <= thetaPhi)
				{
					edges++;
				}
			}
			if ((float)edges < tau * (float)kTpl)
			{
				removedPoints_8u.at<uchar>(y, x) = 255;
			}
		}
	}
	return removedPoints_8u;
}

void filterTemplateEdges(std::vector<DetectionUnit> &templates, int kTpl, float lambda, float thetaD, float thetaPhi, float tau, float removePixelRatio) {
	int totalEdges = 0;
	for (int i = 0; i < templates.size(); i++)
	{
		totalEdges += templates[i].edgesCount;
	}

	float averageEdges = (float)totalEdges / (float)templates.size();
	std::vector<cv::Mat> removedEdges(templates.size());
#pragma omp parallel for
	for (int t = 0; t < templates.size(); t++)
	{
		std::vector<TemplateChamferScore> simmilarity;
		for (int i = 0; i < templates.size(); i++)
		{
			if (i != t)
			{
				float chamferScore = getOrientedChamferScore(templates[t], templates[i], averageEdges, lambda, thetaD, thetaPhi);
				simmilarity.push_back(TemplateChamferScore(i, chamferScore));
			}
		}
		std::sort(simmilarity.begin(), simmilarity.end(), compareDescChamferScore);

		/*showResized("src",  templates[t].edges_8u, 3);
		for (int i = 0; i < 10; i++)
		{
		showResized(std::to_string(i)+"sim", templates[simmilarity[i].index].edges_8u, 3);
		}
		cv::waitKey();*/


		std::vector<DetectionUnit> simmilarTemplates(kTpl);
		for (int i = 0; i < kTpl; i++)
		{
			simmilarTemplates[i] = templates[simmilarity[i].index];
		}
		cv::Mat nonStableEdges = removeNonStablePoints_8u(templates[t], simmilarTemplates, thetaD, thetaPhi, tau);
		removedEdges[t] = removeDiscriminatePoints_8u(templates[t].img_8u, nonStableEdges, removePixelRatio);

		/*showResized("srcEdges", templates[t].edges_8u, 3);
		cv::Mat nonStableEdges = removeNonStablePoints_8u(templates[t], simmilarTemplates, thetaD, thetaPhi, tau);
		showResized("non stable", nonStableEdges, 3);
		removedEdges[t] = removeDiscriminatePoints_8u(templates[t].img_8u, nonStableEdges, removePixelRatio);
		showResized("discrimintaive", removeDiscriminatePoints_8u(templates[t].img_8u, templates[t].edges_8u, removePixelRatio), 3);
		showResized("total", removedEdges[t], 3);
		cv::waitKey(0);*/
		/*if (t % 10 == 0)
		{
		std::printf("Filtered %d / %d templates\r", t, templates.size());
		}*/
	}
#pragma omp parallel for
	for (int t = 0; t < templates.size(); t++) {
		templates[t].edges_8u = removedEdges[t];
		prepareDetectionUnit(templates[t], false, false, true);
	}
}
