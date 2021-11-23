#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct CurveFitData
{
	Mat image, leftPixelK, rightPixelK, leftRealK, rightRealK, curves;
	vector<Point> leftCurvePoints, rightCurvePoints;
	double leftRadius, rightRadius, vehiclePosition;
};

void CurveFit(Mat& in, CurveFitData& outCurveData, double metersPerPixelX, double metersPerPixelY, int numWindows, int windowWidth, int minPixelCount);