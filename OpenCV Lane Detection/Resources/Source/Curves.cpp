#include "Curves.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

CurveFitData CurveFit(Mat& in, int numWindows, int margin, int minPixels)
{
	int windowHeight = in.rows / numWindows;
	int midWidth = in.cols / 2;
	

	Mat lowerHalf = in(Range((float) in.rows / 2, in.rows), Range::all());
	
	Mat hist;
	reduce(lowerHalf, hist, 0, REDUCE_SUM, CV_32F);

	Mat lowerLeft = hist(Range::all(), Range(0, midWidth));
	Mat lowerRight = hist(Range::all(), Range(midWidth, hist.cols));

	double lMin, lMax, rMin, rMax;
	Point lMinP, lMaxP, rMinP, rMaxP;
	minMaxLoc(lowerLeft, &lMin, &lMax, &lMinP, &lMaxP);
	minMaxLoc(lowerRight, &rMin, &rMax, &rMinP, &rMaxP);
	
	lowerHalf.convertTo(lowerHalf, CV_8UC3);
	circle(lowerHalf, Point(lMaxP.x, lowerHalf.rows), 5, Scalar(255, 0, 0));
	circle(lowerHalf, Point(midWidth + rMaxP.x, lowerHalf.rows), 5, Scalar(255, 0, 0));
	cout << lMaxP << endl << rMaxP << endl;

	imshow("Lower Half", lowerHalf);

	return CurveFitData();
}