#pragma once

#include "Curves.h"

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void SkyView(const Mat& in, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints);
void ProjectLane(const Mat& originalIn, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints, CurveFitData curveData, Scalar color = Scalar_(0, 255, 0));