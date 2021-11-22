#pragma once

#include "SkyView.h"

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void SkyView(Mat& in, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints)
{
	Mat warpMatrix = getPerspectiveTransform(sourcePoints, destinationPoints);
	warpPerspective(in, out, warpMatrix, in.size(), INTER_LINEAR);
}

void ProjectLane(Mat& originalIn, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints, CurveFitData curveData, Scalar color)
{
	Mat lane = Mat::zeros(originalIn.size(), CV_8UC3);
	vector<Point> allLanePoints;

	allLanePoints.insert(allLanePoints.end(), curveData.leftCurvePoints.begin(), curveData.leftCurvePoints.end());
	allLanePoints.insert(allLanePoints.end(), curveData.rightCurvePoints.rbegin(), curveData.rightCurvePoints.rend());

	fillPoly(lane, allLanePoints, color);

	Mat warpMatrix = getPerspectiveTransform(sourcePoints, destinationPoints);
	warpPerspective(lane, lane, warpMatrix, lane.size(), INTER_LINEAR);

	out = originalIn.clone();
	addWeighted(out, 1, lane, 0.3, 0, out);
}