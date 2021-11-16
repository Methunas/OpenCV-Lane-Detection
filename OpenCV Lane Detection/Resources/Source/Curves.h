#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct CurveFitData
{
	Mat image;
	double leftRadius;
	double rightRadius;
	Mat leftBestFitCurveReal;
	Mat rightBestFitCurveReal;
	Mat leftBestFitCurvePixel;
	Mat rightBestFitCurvePixel;
	double vehiclePosition;
	string vehicleInformation;
};

CurveFitData CurveFit(Mat& in, int numWindows, int margin, int minPixels);