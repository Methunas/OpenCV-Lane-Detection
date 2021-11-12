#include "Calibration.h"

CalibrationData Calibrate(vector<Mat> images, Size boardSize, float squareSize)
{
	Mat matrix;
	Mat distortion;
	vector<Point2f> corners;
	vector<vector<Point3f>> objectPoints;
	vector<vector<Point2f>> imagePoints;

	for (Mat image : images)
	{
		Mat gray;
		image.convertTo(gray, COLOR_RGB2GRAY);

		bool hasCorners = findChessboardCorners(image, boardSize, corners);

		if (hasCorners)
		{
			cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.1));

			drawChessboardCorners(gray, boardSize, corners, hasCorners);
		}

		vector<Point3f> object;

		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				object.push_back(Point3f(j * squareSize, i * squareSize, 0));

		if (hasCorners)
		{
			imagePoints.push_back(corners);
			objectPoints.push_back(object);
		}
	}

	Mat camMatrix, distortion;
	vector<Mat> rvecs, tvecs;

	calibrateCamera(objectPoints, imagePoints, images[0].size(),
		camMatrix, distortion, rvecs, tvecs);

	return CalibrationData(camMatrix, distortion);
}