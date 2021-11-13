#include "Calibration.h"

#include <iostream>
#include <filesystem>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace filesystem;

int main(int argc, char* argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    string calibrationDirectory;

    for (int i = 0; i < argc; i++)
    {
        string_view arg(argv[i]);

        if (arg == "-c")
            calibrationDirectory = argv[++i];
    }

    vector<Mat> calibrationImages;
    
    for (const auto& file : directory_iterator(calibrationDirectory))
        calibrationImages.push_back(imread(file.path().string()));

    imshow("orig", calibrationImages[0]);

    CalibrationData calibrationData = Calibrate(calibrationImages, Size(9, 6), 1.0);

    Mat undistorted;
    undistort(calibrationImages[0], undistorted, calibrationData.camMatrix, calibrationData.distortion);
    imshow("un", undistorted);

    waitKey(0);
    return 0;
}