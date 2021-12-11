#pragma once

#include "Undistortion.h"
#include "LaneFilter.h"
#include "SkyView.h"
#include "Curves.h"

#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;
using namespace filesystem;

struct FrameData
{
public:
	vector<double> undistortTime, skyViewTime, 
		laneFilterTime, curveFitTime, 
		projectionTime, combineTime,
		leftRadius, rightRadius,
		vehiclePosition;

	void OutputMostRecentToConsole()
	{
		cout << "Undistort Time: " << undistortTime.back() << endl <<
			"Sky View Time: " << skyViewTime.back() << endl <<
			"Lane Filter Time: " << laneFilterTime.back() << endl <<
			"Curve Fit Time: " << curveFitTime.back() << endl <<
			"Projection Time: " << projectionTime.back() << endl <<
			"Combine Time: " << combineTime.back() << endl << endl;
	}

	void OutputToFile(const string path)
	{
		Mat undistortTimeMat(undistortTime),
			skyViewTimeMat(skyViewTime),
			laneFilterTimeMat(laneFilterTime),
			curveFitTimeMat(curveFitTime),
			projectionTimeMat(projectionTime),
			combineTimeMat(combineTime),
			leftRadiusMat(leftRadius),
			rightRadiusMat(rightRadius),
			vehiclePositionMat(vehiclePosition);

		if (!exists(path))
			create_directory(path);

		FileStorage outStream(path + "\\frame_data.yml", FileStorage::WRITE);

		if (!outStream.isOpened())
			return;

		outStream << "Undistort Time" << undistortTimeMat <<
			"Sky View Time" << skyViewTimeMat <<
			"Lane Filter Time" << laneFilterTimeMat <<
			"Curve Fit Time" << curveFitTimeMat <<
			"Projection Time" << projectionTimeMat <<
			"Combine Time" << combineTimeMat <<
			"Left Curve Radius" << leftRadiusMat <<
			"Right Curve Radius" << rightRadiusMat <<
			"Vehicle Position" << vehiclePositionMat;

		outStream.release();
	}
};

void ProcessFrame(Mat& frame, const CalibrationData& calibrationData, const PartUndistortMapData& undistortMapData, FrameData& frameData, bool showStepsInNewWindows, bool combineStepsInFinalFrame);