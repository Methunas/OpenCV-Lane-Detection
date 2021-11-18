#include "Curves.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

Point FindInitialLanePoints(Mat& in, Range heightRange)
{
	Mat lowerWindow = in(heightRange, Range::all());

	Mat hist;
	reduce(lowerWindow, hist, 0, REDUCE_SUM, CV_32F);

	Mat lowerLeft = hist(Range::all(), Range(0, in.cols / 2));
	Mat lowerRight = hist(Range::all(), Range(in.cols / 2, hist.cols));

	double lMin, lMax, rMin, rMax;
	Point lMinP, lMaxP, rMinP, rMaxP;
	minMaxLoc(lowerLeft, &lMin, &lMax, &lMinP, &lMaxP);
	minMaxLoc(lowerRight, &rMin, &rMax, &rMinP, &rMaxP);

	return Point(lMaxP.x, rMaxP.x);
}

int FindWindowLanePoint(Mat& in, Rect bounds, int minPixelCount)
{
	Mat window = in(bounds);

	int pixelCount = countNonZero(window);

	if (pixelCount == 0)
		return bounds.width / 2;

	Mat hist;
	reduce(window, hist, 0, REDUCE_SUM, CV_32F);
	
	float average = 0.00f;

	for (int i = 0; i < hist.cols; i++)
		average += hist.at<float>(i) * i;

	average /= pixelCount;

	return (int)average;
}

void CurveFit(Mat& in, CurveFitData& outCurveData, int numWindows, int windowWidth, int minPixelCount)
{
	int windowHeight = in.rows / numWindows;
	int midImageWidth = in.cols / 2;
	int midWindowWidth = windowWidth / 2;

	outCurveData.image = in.clone();

	Point centers = FindInitialLanePoints(in, Range(in.rows / 2, in.rows));

	int currentLeftX = centers.x;
	int currentRightX = centers.y + midImageWidth;

	Rect leftWindow = Rect(
		currentLeftX - windowWidth / 2,
		in.rows - windowHeight,
		windowWidth, windowHeight);

	Rect rightWindow = Rect(
		currentRightX - windowWidth / 2,
		in.rows - windowHeight,
		windowWidth, windowHeight);
	
	for (int i = 0; i < numWindows; i++)
	{
		leftWindow.y = in.rows - (i + 1) * windowHeight;
		rightWindow.y = in.rows - (i + 1) * windowHeight;

		int lanePointL = FindWindowLanePoint(in, leftWindow, minPixelCount);
		int lanePointR = FindWindowLanePoint(in, rightWindow, minPixelCount);

		currentLeftX += lanePointL - midWindowWidth;
		currentRightX += lanePointR - midWindowWidth;

		leftWindow.x = clamp(currentLeftX - midWindowWidth, 0, in.cols - windowWidth);
		rightWindow.x = clamp(currentRightX - midWindowWidth, 0, in.cols - windowWidth);
		
		rectangle(outCurveData.image, leftWindow, Scalar::all(255));
		rectangle(outCurveData.image, rightWindow, Scalar::all(255));
		cout << endl;
	}
}