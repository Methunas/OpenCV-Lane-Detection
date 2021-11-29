#include "Undistortion.h"

#include <opencv2/imgproc.hpp>

// This function is a modified version of cv::undistort which calculates all of the rectify maps and outputs them
void CalculatePartUndistortMaps(PartUndistortMapData& undistortMapDataOut, const Size imageSize, const CalibrationData& calibrationData)
{
    Mat distCoeffs = calibrationData.distortion, cameraMatrix = calibrationData.camMatrix;

    int stripe_size0 = min(max(1, (1 << 12) / max(imageSize.width, 1)), imageSize.height);
    Mat map1(stripe_size0, imageSize.width, CV_16SC2), map2(stripe_size0, imageSize.width, CV_16UC1);

    Mat_<double> A, Ar, I = Mat_<double>::eye(3, 3);

    cameraMatrix.convertTo(A, CV_64F);
    distCoeffs = Mat_<double>(distCoeffs);

    A.copyTo(Ar);

    double v0 = Ar(1, 2);
    for (int y = 0; y < imageSize.height; y += stripe_size0)
    {
        int stripe_size = min(stripe_size0, imageSize.height - y);
        Ar(1, 2) = v0 - y;
        Mat map1_part = map1.rowRange(0, stripe_size),
            map2_part = map2.rowRange(0, stripe_size);

        initUndistortRectifyMap(A, distCoeffs, I, Ar, Size(imageSize.width, stripe_size),
            map1_part.type(), map1_part, map2_part);

        undistortMapDataOut.map1_parts.push_back(map1_part.clone());
        undistortMapDataOut.map2_parts.push_back(map2_part.clone());
    }
}

// This function is a modified version of cv::undistort which remaps the frame without recalculating the recify maps
void RemapFrame(const Mat& frame, Mat& out, const CalibrationData& calibrationData, const PartUndistortMapData& undistortMapData)
{
    out.create(frame.size(), frame.type());
    int stripe_size0 = std::min(max(1, (1 << 12) / max(frame.cols, 1)), frame.rows);

    for (int y = 0, i = 0; y < frame.rows; y += stripe_size0, i++)
    {
        int stripe_size = min(stripe_size0, frame.rows - y);
        Mat dst_part = out.rowRange(y, y + stripe_size);
        remap(frame, dst_part, undistortMapData.map1_parts.at(i), undistortMapData.map2_parts.at(i), INTER_LINEAR, BORDER_CONSTANT);
    }
}