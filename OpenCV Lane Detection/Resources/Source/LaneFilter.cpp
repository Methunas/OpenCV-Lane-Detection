#include "LaneFilter.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void ColorMask(Mat& in, Mat& out, LaneFilterArgs args)
{
	Mat splitChannels[3];
	
	split(in, splitChannels);

	Mat lightnessThreshold;
	Mat saturationThreshold;
	
	threshold(splitChannels[1], lightnessThreshold, args.lightnessThreshold, 255, THRESH_BINARY);
	threshold(splitChannels[2], saturationThreshold, args.saturationThreshold, 255, THRESH_BINARY);

	Mat threshold1 = lightnessThreshold & saturationThreshold;
	Mat threshold2;

	threshold(splitChannels[1], threshold2, args.lightnessThresholdAgr, 255, THRESH_BINARY);

	threshold1.convertTo(threshold1, CV_8U);
	threshold2.convertTo(threshold2, CV_8U);

	out = threshold1 | threshold2;
}

void SobelMask(Mat& in, Mat& out, LaneFilterArgs args)
{
	Mat splitChannels[3];
	split(in, splitChannels);

	Mat sobelX;
	Mat sobelY;

	Sobel(splitChannels[1], sobelX, CV_64F, 1, 0, 5);
	Sobel(splitChannels[1], sobelY, CV_64F, 0, 1, 5);

	sobelX = abs(sobelX);
	sobelY = abs(sobelY);

	Mat direction = Mat::zeros(Size(in.cols, in.rows), CV_64F);

	for (int i = 0; i < in.rows; i++)
		for (int j = 0; j < in.cols; j++)
			direction.at<double>(i, j) = atan2(abs(sobelY.at<double>(i, j)), abs(sobelX.at<double>(i, j)));

	direction.convertTo(direction, CV_8U);
	
	Mat magnitude;
	Mat squaredSobelX;
	Mat squaredSobelY;

	cv::pow(sobelX, 2, squaredSobelX);
	cv::pow(sobelY, 2, squaredSobelY);

	sqrt(squaredSobelX + squaredSobelY, magnitude);

	double min, max;
	minMaxLoc(magnitude, &min, &max);
	convertScaleAbs(magnitude, magnitude, 255 / max);

	Mat scaledSobelX;

	minMaxLoc(sobelX, &min, &max);
	convertScaleAbs(sobelX, scaledSobelX, 255 / max);

	Mat threshold1;
	Mat threshold2;
	Mat threshold3;
	Mat threshold4;

	threshold(magnitude, threshold1, args.magnitudeThreshold, 255, THRESH_BINARY);
	threshold(scaledSobelX, threshold2, args.xThreshold, 255, THRESH_BINARY);
	threshold(direction, threshold3, args.directionThreshold.x, 255, THRESH_BINARY);
	threshold(direction, threshold4, args.directionThreshold.y, 255, THRESH_BINARY);

	out = threshold1 & threshold2 & threshold3 & ~threshold4;
}

void LaneFilter(Mat& in, LaneFilterData& out, LaneFilterArgs args)
{
	Mat hlsImage;
	in.convertTo(hlsImage, COLOR_RGB2HLS);

	ColorMask(hlsImage, out.colorMask, args);
	SobelMask(hlsImage, out.sobelMask, args);

	out.combinedMask = out.colorMask | out.sobelMask;
}