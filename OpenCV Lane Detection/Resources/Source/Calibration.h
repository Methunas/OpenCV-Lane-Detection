#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace filesystem;

struct CalibrationData
{
	Mat camMatrix, distortion;

	void OutputToFile(string path)
	{
		if (!exists(path))
			create_directory(path);
		
		FileStorage outStream(path + "\\calibration.yml", FileStorage::WRITE);

		if (!outStream.isOpened())
			return;

		outStream << "camMatrix" << camMatrix << "distortion" << distortion;
		outStream.release();
	}

	bool LoadFromFile(string path)
	{
		FileStorage inStream(path + "\\calibration.yml", FileStorage::READ);

		if (!inStream.isOpened())
			return false;

		inStream["camMatrix"] >> camMatrix;
		inStream["distortion"] >> distortion;
		inStream.release();

		return true;
	}
};

CalibrationData Calibrate(std::vector<cv::Mat> images, cv::Size boardSize, float squareSize);