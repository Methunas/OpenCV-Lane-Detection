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

	CalibrationData() = default;
	CalibrationData(Mat camMatrix, Mat distortion) : camMatrix(camMatrix), distortion(distortion) {}
};

class Calibration
{
private:
	vector<Mat> m_images;
	Size m_boardSize;
	Mat m_matrix;
	Mat m_distortion;
	vector<Point2f> m_corners;
	vector<vector<Point3f>> m_objectPoints;
	vector<vector<Point2f>> m_imagePoints;
	float m_squareSize;

public:
	CalibrationData calibrationData;

public:
	Calibration() = default;
	Calibration(vector<Mat> images, Size size, float squareSize);
	CalibrationData Calibrate();
};
