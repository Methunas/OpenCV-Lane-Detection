#include "Calibration.h"

Calibration::Calibration(vector<Mat> images, Size boardSize = Size(9, 6), float squareSize)
	: m_images(images), m_boardSize(boardSize), m_squareSize(squareSize)
{
	
}

CalibrationData Calibration::Calibrate()
{
	for (Mat image : m_images)
	{
		Mat gray;
		image.convertTo(gray, COLOR_RGB2GRAY);

		bool hasCorners = findChessboardCorners(image, m_boardSize, m_corners);

		if (hasCorners)
		{
			cornerSubPix(gray, m_corners, Size(5, 5), Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.1));

			drawChessboardCorners(gray, m_boardSize, m_corners, hasCorners);
		}

		vector<Point3f> object;

		for (int i = 0; i < m_boardSize.height; i++)
			for (int j = 0; j < m_boardSize.width; j++)
				object.push_back(Point3f(j * m_squareSize, i * m_squareSize, 0));

		if (hasCorners)
		{
			m_imagePoints.push_back(m_corners);
			m_objectPoints.push_back(object);
		}
	}

	Mat camMatrix, distortion;
	vector<Mat> rvecs, tvecs;

	calibrateCamera(m_objectPoints, m_imagePoints, m_images[0].size(),
		camMatrix, distortion, rvecs, tvecs);

	return CalibrationData(camMatrix, distortion);
}