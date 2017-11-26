#pragma once

void showResized(cv::String label, cv::Mat img, int ratio, int waitToDraw = -1) {
	cv::Mat toShow;
	if (ratio > 1)
	{
		cv::resize(img, toShow, cv::Size(), ratio, ratio, cv::INTER_AREA);
	}
	else
	{
		toShow = img;
	}
	cv::imshow(label, toShow);
	if (waitToDraw >= 0) {
		cv::waitKey(waitToDraw);
	}
}

// Not used anymore
cv::Mat joinBgWithEdges_8ucv3(cv::Mat src_8u, cv::Mat edges_8u, float edgeRatio = 0.3) {
	cv::Mat src_BGR;
	cv::cvtColor(src_8u, src_BGR, CV_GRAY2BGR);

	for (int x = 0; x < src_8u.cols; x++)
	{
		for (int y = 0; y < src_8u.rows; y++)
		{
			if (edges_8u.at<uchar>(y, x) == 0) {
				src_BGR.at<cv::Vec3b>(y, x) = src_BGR.at<cv::Vec3b>(y, x) * (1 - edgeRatio) + cv::Vec3b(0, 0, 255) * edgeRatio;
			}
		}
	}

	return src_BGR;
}

/// not used anymore
//cv::Mat getTemplateToShow(DetectionUnit &unit) {
//	cv::Mat ret;
//	unit.img_8u.copyTo(ret);
//	cv::cvtColor(ret, ret, CV_GRAY2BGR);
//	for (int x = 0; x < ret.cols; x++)
//	{
//		for (int y = 0; y < ret.rows; y++)
//		{
//			if (unit.edges_8u.at<uchar>(y, x) == 0)
//			{
//				ret.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
//			}
//		}
//	}
//	return ret;
//}

// FIX this
/// TEST func
//int visualizeTriplets(std::vector<Triplet> triplets, int edgeOffset, int pointsDistance, int imageColsRows, int scaleRatio = 10, int wait = 100)
//{
//	cv::Mat netRaw, netSingle, netAll;
//
//	netRaw = cv::Mat(imageColsRows * scaleRatio, imageColsRows * scaleRatio, CV_8UC3);
//	netRaw = cv::Scalar(55, 55, 55);
//	for (int x = edgeOffset; x < imageColsRows; x += pointsDistance)
//	{
//		cv::line(netRaw, cv::Point(x * scaleRatio, 0), cv::Point(x * scaleRatio, netRaw.cols), cv::Scalar(128, 128, 128));
//		cv::line(netRaw, cv::Point(0, x * scaleRatio), cv::Point(netRaw.cols, x * scaleRatio), cv::Scalar(128, 128, 128));
//	}
//	for (int x = edgeOffset; x < imageColsRows; x += pointsDistance)
//	{
//		for (int y = edgeOffset; y < imageColsRows; y += pointsDistance)
//		{
//			cv::circle(netRaw, cv::Point(x * scaleRatio, y * scaleRatio), 4, cv::Scalar(0, 0, 0), -1);
//		}
//	}
//	netRaw.copyTo(netSingle);
//	netRaw.copyTo(netAll);
//	cv::imshow("netAll", netAll);
//	for (int i = 0; i < triplets.size(); i++)
//	{
//		cv::line(netAll, triplets[i].p1 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
//		cv::line(netAll, triplets[i].p3 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
//		cv::circle(netAll, triplets[i].p1 * scaleRatio, 4, cv::Scalar(255, 255, 255), -1);
//		cv::circle(netAll, triplets[i].p2 * scaleRatio, 4, cv::Scalar(255, 255, 255), -1);
//		cv::circle(netAll, triplets[i].p3 * scaleRatio, 4, cv::Scalar(255, 255, 255), -1);
//
//		cv::line(netSingle, triplets[i].p1 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
//		cv::line(netSingle, triplets[i].p3 * scaleRatio, triplets[i].p2 * scaleRatio, cv::Scalar(0, 0, 255));
//		cv::circle(netSingle, triplets[i].p1 * scaleRatio, 4, cv::Scalar(0, 0, 200), -1);
//		cv::circle(netSingle, triplets[i].p2 * scaleRatio, 4, cv::Scalar(0, 200, 0), -1);
//		cv::circle(netSingle, triplets[i].p3 * scaleRatio, 4, cv::Scalar(200, 0, 0), -1);
//
//		cv::imshow("netSingle", netSingle);
//		cv::waitKey(wait);
//		cv::imshow("netAll", netAll);
//
//		for (int x = 0; x < netRaw.cols; x++)
//		{
//			for (int y = 0; y < netRaw.rows; y++)
//			{
//				netSingle.at<cv::Vec3b>(y, x) = netRaw.at<cv::Vec3b>(y, x);
//			}
//		}
//	}
//	cv::waitKey();
//	return 0;
//}

// FIX THIS
/// TEST func
//void visualizeTripletOnEdges(DetectionUnit &unit, Triplet &triplet, TripletValues *tripletValues = NULL, int wait = 0) {
//	int scaleRatio = 10;
//	cv::Mat show;
//	cv::resize(unit.edges_8u, show, cv::Size(), scaleRatio, scaleRatio, cv::INTER_AREA);
//	for (size_t i = 0; i < show.cols * show.rows; i++)
//	{
//		if (show.at<uchar>(i) == 0)
//		{
//			show.at<uchar>(i) = 180;
//		}
//	}
//	cv::cvtColor(show, show, CV_GRAY2BGR);
//
//	cv::Point p1 = triplet.p1 * scaleRatio;
//	cv::Point p2 = triplet.p2 * scaleRatio;
//	cv::Point p3 = triplet.p3 * scaleRatio;
//
//	cv::line(show, p1, p2, cv::Scalar(0, 0, 255), 2);
//	cv::line(show, p3, p2, cv::Scalar(0, 0, 255), 2);
//	cv::circle(show, p1, 4, cv::Scalar(0, 0, 200), -1);
//	cv::circle(show, p2, 4, cv::Scalar(0, 170, 0), -1);
//	cv::circle(show, p3, 4, cv::Scalar(200, 0, 0), -1);
//	if (tripletValues != NULL)
//	{
//		int oriLineLength = 30;
//		cv::line(show, p1, cv::Point(p1.x - round(cos(tripletValues->phi1) * oriLineLength), p1.y - round(sin(tripletValues->phi1) * oriLineLength)), cv::Scalar(0, 178, 191), 2);
//		cv::line(show, p2, cv::Point(p2.x - round(cos(tripletValues->phi2) * oriLineLength), p2.y - round(sin(tripletValues->phi2) * oriLineLength)), cv::Scalar(0, 178, 191), 2);
//		cv::line(show, p3, cv::Point(p3.x - round(cos(tripletValues->phi3) * oriLineLength), p3.y - round(sin(tripletValues->phi3) * oriLineLength)), cv::Scalar(0, 178, 191), 2);
//
//		int textLength = 140;
//		p1.x += 1 * scaleRatio; p2.x += 1 * scaleRatio; p3.x += 1 * scaleRatio;
//		p1.y -= 1 * scaleRatio; p2.y -= 1 * scaleRatio; p3.y -= 1 * scaleRatio;
//		if (p1.x + textLength > show.cols) { p1.x -= textLength; }
//		if (p2.x + textLength > show.cols) { p2.x -= textLength; }
//		if (p3.x + textLength > show.cols) { p3.x -= textLength; }
//
//		std::stringstream ss;
//		ss << std::fixed << std::setprecision(2) << "d: " << tripletValues->d1 << " phi: " << tripletValues->phi1;
//		cv::putText(show, ss.str(), p1, CV_FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(0, 0, 200), 2);
//		ss.str("");
//		ss.clear();
//		ss << std::fixed << std::setprecision(2) << "d: " << tripletValues->d2 << " phi: " << tripletValues->phi2;
//		cv::putText(show, ss.str(), p2, CV_FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(0, 170, 0), 2);
//		ss.str("");
//		ss.clear();
//		ss << std::fixed << std::setprecision(2) << "d: " << tripletValues->d3 << " phi: " << tripletValues->phi3;
//		cv::putText(show, ss.str(), p3, CV_FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(200, 0, 0), 2);
//	}
//
//	cv::imshow("Triplet on edges", show);
//	cv::waitKey(wait);
//}