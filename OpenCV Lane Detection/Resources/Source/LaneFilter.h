#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct LaneFilterArgs
{
	int saturationThreshold;
	int lightnessThreshold;
	int lightnessThresholdAgr;
	Point2f directionThreshold;
	int magnitudeThreshold;
	int xThreshold;
};

struct LaneFilterData
{
	Mat colorMask, sobelMask, combinedMask;
};

void LaneFilter(Mat& in, LaneFilterData& out, LaneFilterArgs args);
