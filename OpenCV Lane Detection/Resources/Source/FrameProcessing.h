#pragma once

#include "Undistortion.h"
#include "LaneFilter.h"
#include "SkyView.h"
#include "Curves.h"

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void ProcessFrame(Mat& frame, const CalibrationData& calibrationData, const PartUndistortMapData& undistortMapData, bool showStepsInNewWindows, bool combineStepsInFinalFrame);