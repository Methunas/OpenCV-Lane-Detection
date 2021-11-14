#include "SkyView.h"

#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void SkyView(Mat& in, Mat& out, vector<Point2f> sourcePoints, vector<Point2f> destinationPoints)
{
	Mat warpMatrix = getPerspectiveTransform(sourcePoints, destinationPoints);
	warpPerspective(in, out, warpMatrix, in.size(), INTER_LINEAR);
}