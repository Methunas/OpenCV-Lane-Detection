#pragma once

#include "SkyView.h"

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void SkyView(const Mat& in, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints)
{
	// Warp the image into sky view using the specified points
	Mat warpMatrix = getPerspectiveTransform(sourcePoints, destinationPoints);
	warpPerspective(in, out, warpMatrix, in.size(), INTER_LINEAR);
}

void ProjectLane(const Mat& originalIn, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints, CurveFitData curveData, Scalar color)
{
	Mat lane = Mat::zeros(originalIn.size(), CV_8UC3);

	// Create a new vector that outlines the bounds of the polygon (clockwise from the bottom left) and fill the shape
	vector<Point> allLanePoints;

	allLanePoints.insert(allLanePoints.end(), curveData.leftCurvePoints.begin(), curveData.leftCurvePoints.end());
	allLanePoints.insert(allLanePoints.end(), curveData.rightCurvePoints.rbegin(), curveData.rightCurvePoints.rend());

	fillPoly(lane, allLanePoints, color);

	// Undo the sky view and add the lane to the original image
	Mat warpMatrix = getPerspectiveTransform(sourcePoints, destinationPoints);
	warpPerspective(lane, lane, warpMatrix, lane.size(), INTER_LINEAR);

	out = originalIn.clone();
	addWeighted(out, 1, lane, 0.3, 0, out);
}