#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct LaneFilterArgs
{
	int saturationThreshold;
	int lightThreshold;
	int lightThresholdAgr;
	Point2i gradientThreshold;
	int magnificationThreshold;
	int xThreshold;
};

void ApplyLaneFilter(Mat in, Mat out, LaneFilterArgs args);
void ApplyColorMask(Mat frame);
void ApplySobelMask(Mat frame);
