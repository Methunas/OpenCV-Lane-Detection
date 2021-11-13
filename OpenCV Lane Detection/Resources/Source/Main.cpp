#include "Calibration.h"

#include <iostream>
#include <filesystem>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace filesystem;

struct UndistortMapData
{
    Mat map1, map2;
};

void ProcessFrame(Mat frame, CalibrationData calibrationData, UndistortMapData undistortMapData)
{
    #pragma region Undistort

    remap(frame.clone(), frame, undistortMapData.map1, undistortMapData.map2, INTER_LINEAR, BORDER_CONSTANT);
    Mat cropped = frame(Range(20, frame.rows - 20), Range(20, frame.cols - 20));
    frame = cropped.clone();

    #pragma endregion

    #pragma region Birds Eye

    vector<Point> sourcePoints({ {580, 460}, {205, 720}, {1110, 720}, {703, 460} });
    vector<Point> destinationPoints({ {320, 0}, {320, 720}, {960, 720}, {960, 0} });

    for (Point point : sourcePoints)
        circle(frame, point, 4, Scalar(0,0,255), FILLED);

    #pragma endregion
}

int main(int argc, char* argv[])
{
    #pragma region Argument Parse

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    string calibrationDirectory;
    bool calibrationDataFile = false;
    string videoPath;

    for (int i = 0; i < argc; i++)
    {
        string_view arg(argv[i]);

        if (arg == "-c")
            calibrationDirectory = argv[++i];
        if (arg == "-d")
            calibrationDataFile = true;
        if (arg == "-v")
            videoPath = argv[++i];
    }

    #pragma endregion

    #pragma region Calibration

    CalibrationData calibrationData;

    if (!calibrationData.LoadFromFile("Resources\\SaveData", "calibration.json"))
    {
        vector<Mat> calibrationImages;

        for (const auto& file : directory_iterator(calibrationDirectory))
            calibrationImages.push_back(imread(file.path().string()));

        calibrationData = Calibrate(calibrationImages, Size(9, 6), 1.0);

        calibrationData.OutputToFile("Resources\\SaveData", "calibration.json");
    }

    Mat undistorted = imread("Resources\\Images\\Calibration\\calibration2.jpg");
    undistort(undistorted.clone(), undistorted, calibrationData.camMatrix, calibrationData.distortion);
    imshow("un", undistorted);

    #pragma endregion

    VideoCapture video(videoPath);
    assert(video.isOpened());

    double fps = video.get(VideoCaptureProperties::CAP_PROP_FPS);
    int frameDelay = (int)(1000 / fps);
    Size videoSize((int)video.get(cv::CAP_PROP_FRAME_WIDTH), (int)video.get(cv::CAP_PROP_FRAME_HEIGHT));

    UndistortMapData undistortMapData;
    initUndistortRectifyMap(calibrationData.camMatrix, calibrationData.distortion, Mat(),
        getOptimalNewCameraMatrix(calibrationData.camMatrix, calibrationData.distortion, videoSize, 1, videoSize, 0), 
        videoSize, CV_16SC2, undistortMapData.map1, undistortMapData.map2);

    for (;;)
    {
        Mat frame;
        video >> frame;

        if (frame.empty())
        {
            video.set(VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        imshow("Lane Detection Distorted", frame);
        ProcessFrame(frame, calibrationData, undistortMapData);

        imshow("Lane Detection Undistorted", frame);

        waitKey(frameDelay);
    }

    waitKey(0);
    return 0;
}