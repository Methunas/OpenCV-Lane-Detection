#pragma once
#include "json.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
using namespace filesystem;
using json = nlohmann::json;

typedef struct CalibrationData
{
	Mat camMatrix, distortion;

	void OutputToFile(string path, string fileName)
	{
		vector<float> camMatrixData;
		for (int i = 0; i < camMatrix.rows; ++i)
			camMatrixData.insert(camMatrixData.end(), camMatrix.ptr<float>(i), camMatrix.ptr<float>(i) + camMatrix.cols * camMatrix.channels());

		vector<float> distortionData;
		distortionData.assign(distortion.data, distortion.data + distortion.total() * distortion.channels());

		json contents = {
			{"camMatrix", camMatrixData},
			{"distortion", distortionData}
		};
		
		if (!exists(path))
			create_directory(path);

		ofstream outStream(path + '\\' + fileName);

		if (outStream.fail())
		{
			cout << "Failed to create output file." << endl;
			return;
		}

		outStream << contents;
	}

	bool LoadFromFile(string path, string fileName)
	{
		ifstream inStream(path + '\\' + fileName);

		if (inStream.fail())
			return false;

		json contents;
		inStream >> contents;

		vector<float> camMatrixData = contents["camMatrix"];
		camMatrix = Mat(3, 3, CV_32F, camMatrixData.data());

		vector<float> distortionData = contents["distortion"];
		distortion = Mat(5, 1, CV_32F, distortionData.data());
	}
};

CalibrationData Calibrate(std::vector<cv::Mat> images, cv::Size boardSize, float squareSize);