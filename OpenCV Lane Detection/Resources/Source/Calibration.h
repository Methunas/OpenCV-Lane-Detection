#pragma once
#include "json.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>

using namespace std;
using namespace cv;
using namespace filesystem;
using json = nlohmann::json;

struct CalibrationData
{
	Mat camMatrix, distortion;

	void OutputToFile(string path, string fileName)
	{
		vector<double> camMatrixData;
		for (int i = 0; i < camMatrix.rows; i++)
			for (int j = 0; j < camMatrix.cols; j++)
				camMatrixData.push_back(camMatrix.at<double>(i, j));

		vector<double> distortionData;
		distortionData.assign(distortion.data, distortion.data + distortion.total() * distortion.channels());

		json contents = {
			{"camMatrix", camMatrixData},
			{"distortion", distortionData}
		};
		
		if (!exists(path))
			create_directory(path);
		
		ofstream outStream(path + '\\' + fileName);

		if (outStream.fail())
			return;

		outStream << contents;
	}

	bool LoadFromFile(string path, string fileName)
	{
		ifstream inStream(path + '\\' + fileName);

		if (inStream.fail())
			return false;

		json contents;
		inStream >> contents;

		vector<double> camMatrixData = contents["camMatrix"];
		camMatrix = Mat::zeros(Size(3, 3), CV_64F);
		for (int i = 0; i < camMatrixData.size(); i++)
			camMatrix.at<double>((i / 3), i % 3) = camMatrixData[i];

		vector<double> distortionData = contents["distortion"];
		distortion = Mat(5, 1, CV_64F, distortionData.data());

		return true;
	}
};

CalibrationData Calibrate(std::vector<cv::Mat> images, cv::Size boardSize, float squareSize);