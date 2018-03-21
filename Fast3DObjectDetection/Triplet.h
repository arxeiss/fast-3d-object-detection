#pragma once
class Triplet
{
public:
	cv::Point p1, p2, p3;

	Triplet(cv::Point &p1, cv::Point &p2, cv::Point &p3) {
		this->p1 = p1;
		this->p2 = p2;
		this->p3 = p3;
	}

	Triplet(int inCols, int p1, int p2, int p3) {
		this->p1 = cv::Point(p1 % inCols, p1 / inCols);
		this->p2 = cv::Point(p2 % inCols, p2 / inCols);
		this->p3 = cv::Point(p3 % inCols, p3 / inCols);
	}

	bool operator==(const Triplet&comp) const {
		return this->p2 == comp.p2 &&
			(( this->p1 == comp.p1 && this->p3 == comp.p3 ) || 
			(this->p3 == comp.p1 && this->p1 == comp.p3));
	}
	//~Triplet();
};

