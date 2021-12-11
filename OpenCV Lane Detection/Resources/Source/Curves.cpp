#include "Curves.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

Point FindInitialLanePoints(const Mat& in, const Range heightRange)
{
	Mat lowerWindow = in(heightRange, Range::all());

	Mat hist;
	reduce(lowerWindow, hist, 0, REDUCE_SUM, CV_32F);

	Mat lowerLeft = hist(Range::all(), Range(0, in.cols / 2));
	Mat lowerRight = hist(Range::all(), Range(in.cols / 2, hist.cols));

	float lMax = 0, rMax = 0;
	int lMaxIndex, rMaxIndex;

	for (int i = 0; i < lowerLeft.cols; i++)
	{
		float val = lowerLeft.at<float>(i);

		if (lMax < val)
		{
			lMax = val;
			lMaxIndex = i;
		}
	}

	for (int i = 0; i < lowerRight.cols; i++)
	{
		float val = lowerRight.at<float>(i);

		if (rMax < val)
		{
			rMax = val;
			rMaxIndex = i;
		}
	}

	return Point(lMaxIndex, rMaxIndex);
}

int FindWindowLanePoint(const Mat& window, Rect bounds, int minPixelCount)
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

// Polynomial fit function adapted from:
// https://windowsquestions.com/2021/07/07/polynomial-curve-fitting-in-opencv-c/
Mat PolynomialFit(const vector<Point>& points, int order)
{
	cv::Mat U(points.size(), (order + 1), CV_32F);
	cv::Mat Y(points.size(), 1, CV_32F);

	for (int i = 0; i < U.rows; i++)
	{
		float* rowPtr = U.ptr<float>(i);

		for (int j = 0; j < U.cols; j++)
			rowPtr[j] = pow(points[i].y, j);
	}

	for (int i = 0; i < Y.rows; i++)
	{
		float* rowPtr = Y.ptr<float>(i);
		rowPtr[0] = points[i].x;
	}

	cv::Mat K((order + 1), 1, CV_32F);

	if (U.data != NULL)
		K = (U.t() * U).inv() * U.t() * Y;

	return K;
}

Mat PolynomialFit(const vector<Point2d>& points, int order)
{
	cv::Mat U(points.size(), (order + 1), CV_32F);
	cv::Mat Y(points.size(), 1, CV_32F);

	for (int i = 0; i < U.rows; i++)
	{
		float* rowPtr = U.ptr<float>(i);

		for (int j = 0; j < U.cols; j++)
			rowPtr[j] = pow(points[i].y, j);
	}

	for (int i = 0; i < Y.rows; i++)
	{
		float* rowPtr = Y.ptr<float>(i);
		rowPtr[0] = points[i].x;
	}

	cv::Mat K((order + 1), 1, CV_32F);

	if (U.data != NULL)
		K = (U.t() * U).inv() * U.t() * Y;

	return K;
}

vector<Point> GetCurvePoints(const Mat& K, const vector<Point>& points, int rows, int order)
{
	vector<Point> curvePoints;
	
	for (int j = 0; j < rows; j++)
	{
		Point2d point(0, j);

		for (int k = 0; k < order + 1; k++)
		{
			const float* rowPtr = K.ptr<float>(k);
			point.x += rowPtr[0] * pow(j, k);
		}

		curvePoints.push_back(point);
	}

	return curvePoints;
}

double GetRadiusOfCurvature(Mat K, float y)
{
	// ay^2 + by + c
	float a = K.at<float>(2);
	float b = K.at<float>(1);

	return pow(1 + pow(2 * a * y + b, 2), 1.5) / abs(2 * a);
}

void GetVehiclePosition(CurveFitData& data, float metersPerPixel)
{
	int midWidth = data.image.cols / 2;
	
	// Get the points at the bottom of the frame
	int leftPoint = data.leftCurvePoints[data.leftCurvePoints.size() - 1].x;
	int rightPoint = data.rightCurvePoints[data.leftCurvePoints.size() - 1].x;

	float pixelPosition = leftPoint + (rightPoint - leftPoint) / 2.0;
	
	data.vehiclePosition = (pixelPosition - midWidth) * metersPerPixel;
}

void CurveFit(const Mat& in, CurveFitData& outCurveData, float metersPerPixelX, float metersPerPixelY, int numWindows, int windowWidth, int minPixelCount)
{
	int windowHeight = in.rows / numWindows;
	int midImageWidth = in.cols / 2;
	int midWindowWidth = windowWidth / 2;

	outCurveData.image = in.clone() * 255;

	// Find a good starting position for the bottom most window
	Point centers = FindInitialLanePoints(in, Range(in.rows / 2, in.rows));

	// Define first window bounds
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
	
	// For each of the windows, find the average x position and add reposition the window at thaat point
	// Add the pixels in the window to a separate image, storing one lane line
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
		
		rectangle(outCurveData.image, leftWindowBounds, Scalar::all(255), 3);
		rectangle(outCurveData.image, rightWindowBounds, Scalar::all(255), 3);

		leftWindow.copyTo(leftLane(leftWindowBounds));
		rightWindow.copyTo(rightLane(rightWindowBounds));
	}

	// Find the lane pixel positions and curve fit them separately
	findNonZero(leftLane, leftLanePixels);
	findNonZero(rightLane, rightLanePixels);

	outCurveData.leftPixelK = PolynomialFit(leftLanePixels, 2);
	outCurveData.rightPixelK = PolynomialFit(rightLanePixels, 2);

	outCurveData.leftCurvePoints = GetCurvePoints(outCurveData.leftPixelK, leftLanePixels, in.rows, 2);
	outCurveData.rightCurvePoints = GetCurvePoints(outCurveData.rightPixelK, rightLanePixels, in.rows, 2);

	// Get the vehicle offset from the center of the lane
	GetVehiclePosition(outCurveData, metersPerPixelX);

	// Scale the pixel positions in terms of meters per pixel
	vector<Point2d> leftLaneRealPoints;
	vector<Point2d> rightLaneRealPoints;

	for (Point2d point : leftLanePixels)
		leftLaneRealPoints.emplace_back(point.x * metersPerPixelX, point.y * metersPerPixelY);

	for (Point2d point : rightLanePixels)
		rightLaneRealPoints.emplace_back(point.x * metersPerPixelX, point.y * metersPerPixelY);

	outCurveData.leftRealK = PolynomialFit(leftLaneRealPoints, 2);
	outCurveData.rightRealK = PolynomialFit(rightLaneRealPoints, 2);

	outCurveData.leftRadius = GetRadiusOfCurvature(outCurveData.leftRealK, in.cols * metersPerPixelX);
	outCurveData.rightRadius = GetRadiusOfCurvature(outCurveData.rightRealK, in.cols * metersPerPixelX);

	// Draw the curves and convert color from gray to bgr
	cvtColor(outCurveData.image, outCurveData.image, COLOR_GRAY2BGR);

	for (Point point : outCurveData.leftCurvePoints)
		circle(outCurveData.image, point, 3, cv::Scalar(255, 0, 255), -1, LineTypes::LINE_AA);

	for (Point point : outCurveData.rightCurvePoints)
		circle(outCurveData.image, point, 3, cv::Scalar(255, 0, 255), -1, LineTypes::LINE_AA);
}