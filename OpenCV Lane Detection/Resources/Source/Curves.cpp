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

int FindWindowLanePoint(Mat& window, Rect bounds, int minPixelCount)
{
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

	Rect leftWindowBounds = Rect(
		currentLeftX - windowWidth / 2,
		in.rows - windowHeight,
		windowWidth, windowHeight);

	Rect rightWindowBounds = Rect(
		currentRightX - windowWidth / 2,
		in.rows - windowHeight,
		windowWidth, windowHeight);

	vector<Point> leftLanePixels;
	vector<Point> rightLanePixels;
	
	for (int i = 0; i < numWindows; i++)
	{
		leftWindowBounds.y = in.rows - (i + 1) * windowHeight;
		rightWindowBounds.y = in.rows - (i + 1) * windowHeight;

		Mat leftWindow = in(leftWindowBounds);
		Mat rightWindow = in(rightWindowBounds);

		int lanePointL = FindWindowLanePoint(leftWindow, leftWindowBounds, minPixelCount);
		int lanePointR = FindWindowLanePoint(rightWindow, rightWindowBounds, minPixelCount);

		currentLeftX += lanePointL - midWindowWidth;
		currentRightX += lanePointR - midWindowWidth;

		leftWindowBounds.x = clamp(currentLeftX - midWindowWidth, 0, in.cols - windowWidth);
		rightWindowBounds.x = clamp(currentRightX - midWindowWidth, 0, in.cols - windowWidth);
		
		rectangle(outCurveData.image, leftWindowBounds, Scalar::all(255));
		rectangle(outCurveData.image, rightWindowBounds, Scalar::all(255));

		vector<Point> leftWindowPixels;
		vector<Point> rightWindowPixels;

		findNonZero(leftWindow, leftWindowPixels);
		findNonZero(rightWindow, rightWindowPixels);

		for (Point point : leftWindowPixels)
		{
			point.x += leftWindowBounds.x;
			point.y += leftWindowBounds.y;
		}

		for (Point point : rightWindowPixels)
		{
			point.x += rightWindowBounds.x;
			point.y += rightWindowBounds.y;
		}

		leftLanePixels.insert(leftLanePixels.end(), leftWindowPixels.begin(), leftWindowPixels.end());
		rightLanePixels.insert(rightLanePixels.end(), rightWindowPixels.begin(), rightWindowPixels.end());
	}

	
}