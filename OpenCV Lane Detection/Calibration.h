#pragma once
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

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
	Calibration(vector<Mat> images, Size size, float squareSize);
	CalibrationData Calibrate();
};

struct CalibrationData
{
	Mat camMatrix, distortion;

	CalibrationData(Mat camMatrix, Mat distortion)
		: camMatrix(camMatrix), distortion(distortion)
	{

	}
};