#include "LaneFilter.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void ColorMask(const Mat& in, Mat& out, LaneFilterArgs args)
{
	Mat splitChannels[3];
	
	split(in, splitChannels);

	Mat lightnessThreshold;
	Mat saturationThreshold;
	
	// Use a specified threshold value to separate the lane lines by lightness and saturation
	threshold(splitChannels[1], lightnessThreshold, args.lightnessThreshold, 255, THRESH_BINARY);
	threshold(splitChannels[2], saturationThreshold, args.saturationThreshold, 255, THRESH_BINARY);

	Mat threshold1 = lightnessThreshold & saturationThreshold;
	Mat threshold2;

	threshold(splitChannels[1], threshold2, args.lightnessThresholdAgr, 255, THRESH_BINARY);

	threshold1.convertTo(threshold1, CV_8U);
	threshold2.convertTo(threshold2, CV_8U);

	// Combine the masks to include as much lane information as possible
	out = threshold1 | threshold2;
}

void SobelMask(const Mat& in, Mat& out, LaneFilterArgs args)
{
	Mat splitChannels[3];
	split(in, splitChannels);

	Mat sobelX, sobelY;
	
	// Apply a sobel filter to find the gradient of the image (edge detection)
	Sobel(splitChannels[1], sobelX, CV_32F, 1, 0, 5);
	Sobel(splitChannels[1], sobelY, CV_32F, 0, 1, 5);

	sobelX = abs(sobelX);
	sobelY = abs(sobelY);

	// Find the direction of each pixel
	Mat direction = Mat::zeros(Size(in.cols, in.rows), CV_32F);

	for (int i = 0; i < direction.rows; i++)
	{
		float* dirRowPtr = direction.ptr<float>(i);
		float* sobelXRowPtr = sobelX.ptr<float>(i);
		float* sobelYRowPtr = sobelY.ptr<float>(i);

		for (int j = 0; j < direction.cols; j++)
			dirRowPtr[j] = atan2(abs(sobelXRowPtr[j]), abs(sobelYRowPtr[j]));
	}
	
	direction.convertTo(direction, CV_8U);
	
	// Find the magnitude of each pixel
	Mat magnitude, squaredSobelX, squaredSobelY;

	cv::pow(sobelX, 2, squaredSobelX);
	cv::pow(sobelY, 2, squaredSobelY);

	sqrt(squaredSobelX + squaredSobelY, magnitude);

	float max = 0;
	for (int i = 0; i < magnitude.rows; i++)
	{
		const float* rowPtr = magnitude.ptr<float>(i);

		for (int j = 0; j < magnitude.cols; j++)
		{
			float val = rowPtr[j];
			max = std::max(max, val);
		}
	}
		

	convertScaleAbs(magnitude, magnitude, 255 / max);
	Mat scaledSobelX;

	max = 0;
	for (int i = 0; i < sobelX.rows; i++)
	{
		const float* rowPtr = sobelX.ptr<float>(i);

		for (int j = 0; j < sobelX.cols; j++)
		{
			float val = rowPtr[j];
			max = std::max(max, val);
		}
	}
	
	convertScaleAbs(sobelX, scaledSobelX, 255 / max);

	// Threshold each mask and only include pixels present in every image
	Mat threshold1; // Filter by magnitude
	Mat threshold2; // Filter by edge on the x axis
	Mat threshold3; // Filter by minimum angle
	Mat threshold4; // Filter by maximum angle

	threshold(magnitude, threshold1, args.magnitudeThreshold, 255, THRESH_BINARY);
	threshold(scaledSobelX, threshold2, args.xThreshold, 255, THRESH_BINARY);
	threshold(direction, threshold3, args.directionThreshold.x, 255, THRESH_BINARY);
	threshold(direction, threshold4, args.directionThreshold.y, 255, THRESH_BINARY);
	
	out = threshold1 & threshold2 & threshold3 & ~threshold4;
}

void LaneFilter(const Mat& in, LaneFilterData& out, LaneFilterArgs args)
{
	// Convert the image to HLS color space and apply masks to filter out the lane lines
	Mat hlsImage;
	in.convertTo(hlsImage, COLOR_RGB2HLS);

	ColorMask(hlsImage, out.colorMask, args);
	SobelMask(hlsImage, out.sobelMask, args);

	out.combinedMask = out.colorMask | out.sobelMask;
}