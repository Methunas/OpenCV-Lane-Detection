#pragma once

#include "Calibration.h"

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

struct PartUndistortMapData
{
    vector<Mat> map1_parts, map2_parts;
};

void CalculatePartUndistortMaps(PartUndistortMapData& undistortMapDataOut, const Size imageSize, const CalibrationData& calibrationData);
void RemapFrame(const Mat& frame, Mat& out, const CalibrationData& calibrationData, const PartUndistortMapData& undistortMapData);