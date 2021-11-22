#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct CurveFitData
{
	Mat image, leftPixelK, rightPixelK;
	vector<Point> leftCurvePoints, rightCurvePoints;
	double vehiclePosition;
};

void CurveFit(Mat& in, CurveFitData& outCurveData, double metersPerPixel, int numWindows, int windowWidth, int minPixelCount);