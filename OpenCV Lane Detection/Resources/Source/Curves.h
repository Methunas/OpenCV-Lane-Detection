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

void CurveFit(Mat& in, CurveFitData& outCurveData, int numWindows, int windowWidth, int minPixelCount);