#pragma once

#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void SkyView(Mat& in, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints);