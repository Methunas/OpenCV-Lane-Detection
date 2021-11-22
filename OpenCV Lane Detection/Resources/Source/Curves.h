#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct CurveFitData
{
	Mat image, leftPixelK, rightPixelK;
	vector<Point> leftCurvePoints, rightCurvePoints;
};

void CurveFit(Mat& in, CurveFitData& outCurveData, int numWindows, int windowWidth, int minPixelCount);