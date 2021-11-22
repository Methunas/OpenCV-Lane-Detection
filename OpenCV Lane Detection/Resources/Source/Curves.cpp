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

Mat PolynomialFit(vector<Point>& points, int order)
{
	cv::Mat U(points.size(), (order + 1), CV_64F);
	cv::Mat Y(points.size(), 1, CV_64F);

	for (int i = 0; i < U.rows; i++)
		for (int j = 0; j < U.cols; j++)
			U.at<double>(i, j) = pow(points[i].y, j);

	for (int i = 0; i < Y.rows; i++)
		Y.at<double>(i, 0) = points[i].x;

	cv::Mat K((order + 1), 1, CV_64F);

	if (U.data != NULL)
		K = (U.t() * U).inv() * U.t() * Y;

	return K;
}

vector<Point> GetCurvePoints(Mat& K, vector<Point>& points, int rows, int order)
{
	vector<Point> curvePoints;
	
	for (int j = 0; j < rows; j++)
	{
		Point2d point(0, j);

		for (int k = 0; k < order + 1; k++)
			point.x += K.at<double>(k, 0) * pow(j, k);

		curvePoints.push_back(point);
	}

	return curvePoints;
}

void CurveFit(Mat& in, CurveFitData& outCurveData, int numWindows, int windowWidth, int minPixelCount)
{
	int windowHeight = in.rows / numWindows;
	int midImageWidth = in.cols / 2;
	int midWindowWidth = windowWidth / 2;

	outCurveData.image = in.clone() * 255;

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

	Mat leftLane = Mat::zeros(Size(in.cols, in.rows), CV_8U);
	Mat rightLane = Mat::zeros(Size(in.cols, in.rows), CV_8U);
	
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

		leftWindow = in(leftWindowBounds);
		rightWindow = in(rightWindowBounds);
		
		rectangle(outCurveData.image, leftWindowBounds, Scalar::all(255));
		rectangle(outCurveData.image, rightWindowBounds, Scalar::all(255));

		vector<Point> leftWindowPixels;
		vector<Point> rightWindowPixels;

		leftWindow.copyTo(leftLane(leftWindowBounds));
		rightWindow.copyTo(rightLane(rightWindowBounds));
	}

	Mat polyFit = Mat::zeros(Size(leftLane.cols, leftLane.rows), CV_8U);

	findNonZero(leftLane, leftLanePixels);
	findNonZero(rightLane, rightLanePixels);

	outCurveData.leftPixelK = PolynomialFit(leftLanePixels, 2);
	outCurveData.rightPixelK = PolynomialFit(rightLanePixels, 2);

	outCurveData.leftCurvePoints = GetCurvePoints(outCurveData.leftPixelK, leftLanePixels, polyFit.rows, 2);
	outCurveData.rightCurvePoints = GetCurvePoints(outCurveData.rightPixelK, rightLanePixels, polyFit.rows, 2);
	/*
	for (Point point : outCurveData.leftCurvePoints)
		circle(polyFit, point, 1, cv::Scalar(255, 255, 255), -1, LineTypes::LINE_AA);

	for (Point point : outCurveData.rightCurvePoints)
		circle(polyFit, point, 1, cv::Scalar(255, 255, 255), -1, LineTypes::LINE_AA);
	*/
}