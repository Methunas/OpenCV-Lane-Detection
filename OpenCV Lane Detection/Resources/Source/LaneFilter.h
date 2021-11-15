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

void LaneFilter(Mat& in, Mat& out, LaneFilterArgs args);
void ColorMask(Mat& in, Mat& out, LaneFilterArgs args);
void SobelMask(Mat& in, Mat& out, LaneFilterArgs args);
