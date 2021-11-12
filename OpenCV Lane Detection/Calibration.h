#pragma once
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

typedef struct CalibrationData
{
	Mat camMatrix, distortion;
	vector<Mat> rotationVecs, transformationVecs;
	vector<vector<Point3f>> objectPoints;
	vector<vector<Point2f>> imagePoints;
};

CalibrationData Calibrate(vector<Mat> images, Size boardSize, float squareSize);
double GetCalibrationError(CalibrationData calibrationData);