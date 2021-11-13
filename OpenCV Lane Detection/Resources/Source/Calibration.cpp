#include "Calibration.h"

CalibrationData Calibrate(vector<Mat> images, Size boardSize, float squareSize)
{
	if (images.empty())
		return CalibrationData();

	CalibrationData calibrationData;
	vector<Point2f> corners;
	vector<vector<Point3f>> objectPoints;
	vector<vector<Point2f>> imagePoints;
	vector<Mat> rotationVecs, transformationVecs;

	for (Mat image : images)
	{
		// For some reason, OpenCV doesn't want to convert the image to gray
		// It instead converts to 16FC3
		//Mat gray;
		//image.convertTo(gray, COLOR_RGB2GRAY);
		Mat duplicate;
		image.copyTo(duplicate);

		bool hasCorners = findChessboardCorners(image, boardSize, corners);

		if (hasCorners)
		{
			//cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
			//	TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 3, 0.1));

			drawChessboardCorners(duplicate, boardSize, corners, hasCorners);
			//imshow("Corners", duplicate);

			vector<Point3f> object;

			for (int i = 0; i < boardSize.height; i++)
				for (int j = 0; j < boardSize.width; j++)
					object.push_back(Point3f(j * squareSize, i * squareSize, 0));

			imagePoints.push_back(corners);
			objectPoints.push_back(object);
		}
	}

	calibrateCamera(objectPoints, imagePoints, images[0].size(), calibrationData.camMatrix, 
		calibrationData.distortion, rotationVecs, transformationVecs);

	return calibrationData;
}
