#include "Calibration.h"

CalibrationData Calibrate(vector<Mat> images, Size boardSize, float squareSize)
{
	CalibrationData calibrationData;
	vector<Point2f> corners;

	for (Mat image : images)
	{
		Mat gray;
		image.convertTo(gray, COLOR_RGB2GRAY);

		bool hasCorners = findChessboardCorners(image, boardSize, corners);

		if (hasCorners)
		{
			cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.1));

			//drawChessboardCorners(gray, boardSize, corners, hasCorners);
			//imshow("Corners", gray);
		}

		vector<Point3f> object;

		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				object.push_back(Point3f(j * squareSize, i * squareSize, 0));

		if (hasCorners)
		{
			calibrationData.imagePoints.push_back(corners);
			calibrationData.objectPoints.push_back(object);
		}
	}

	calibrateCamera(calibrationData.objectPoints, calibrationData.imagePoints, images[0].size(), calibrationData.camMatrix, 
		calibrationData.distortion, calibrationData.rotationVecs, calibrationData.transformationVecs);

	return calibrationData;
}

double GetCalibrationError(CalibrationData calibrationData)
{
	int totalPoints = 0;
	double totalError = 0;

	for (int i = 0; i < calibrationData.objectPoints.size(); i++)
	{
		vector<Point2f> imagePoints;

		projectPoints(Mat(calibrationData.objectPoints[i]), calibrationData.rotationVecs[i], 
			calibrationData.transformationVecs[i], calibrationData.camMatrix, calibrationData.distortion, imagePoints);

		double error = norm(Mat(calibrationData.imagePoints[i]), Mat(imagePoints), NORM_L2);
		totalError += error * error;
		totalPoints += calibrationData.objectPoints[i].size();
	}
	
	return sqrt(totalError / totalPoints);
}
