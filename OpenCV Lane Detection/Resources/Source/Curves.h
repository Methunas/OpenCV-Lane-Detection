#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct CurveFitData
{
	Mat image, leftPixelK, rightPixelK, leftRealK, rightRealK;
	vector<Point> leftCurvePoints, rightCurvePoints;
	float leftRadius, rightRadius, vehiclePosition;
};

void CurveFit(const Mat& in, CurveFitData& outCurveData, float metersPerPixelX, float metersPerPixelY, int numWindows, int windowWidth, int minPixelCount);