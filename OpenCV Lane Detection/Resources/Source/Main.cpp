#include "Calibration.h"
#include "LaneFilter.h"
#include "SkyView.h"
#include "Curves.h"

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

void ProcessFrame(Mat& frame, CalibrationData& calibrationData, UndistortMapData& undistortMapData)
{
    #pragma region Undistort

    undistort(frame.clone(), frame, calibrationData.camMatrix, calibrationData.distortion);
    //remap(frame.clone(), frame, undistortMapData.map1, undistortMapData.map2, INTER_LINEAR, BORDER_CONSTANT);

    #pragma endregion

    #pragma region Sky View

    vector<Point2f> sourcePoints({ {580, 460}, {205, 720}, {1110, 720}, {703, 460} });
    vector<Point2f> destinationPoints({ {320, 0}, {320, 720}, {960, 720}, {960, 0} });

    Mat skyView;
    SkyView(frame, skyView, sourcePoints, destinationPoints);
    imshow("Sky View", skyView);

    #pragma endregion

    #pragma region Lane Filter

    Mat laneFilter;
    LaneFilterArgs laneFilterArgs(220, 40, 205, Point2f(0.7f, 1.4f), 40, 20);

    LaneFilter(frame, laneFilter, laneFilterArgs);
    SkyView(laneFilter, laneFilter, sourcePoints, destinationPoints);

    rectangle(laneFilter, Point(0, 0), Point(125, laneFilter.rows), Scalar(0, 0, 0), -1);
    rectangle(laneFilter, Point(laneFilter.cols, 0), Point(1200, laneFilter.rows), Scalar(0, 0, 0), -1);

    imshow("Lane Filter", laneFilter);

    #pragma endregion

    #pragma region Curve Fitting

    Mat binary;
    threshold(laneFilter, binary, 150, 1, THRESH_BINARY);

    CurveFitData curveData;
    CurveFit(binary, curveData, 9, 200, 50);
    imshow("Curve Fitting", curveData.image * 255);

    #pragma endregion
}

int main(int argc, char* argv[])
{
    #pragma region Argument Parse

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    string calibrationDirectory;
    string videoPath;

    for (int i = 0; i < argc; i++)
    {
        string_view arg(argv[i]);

        if (arg == "-c")
            calibrationDirectory = argv[++i];
        if (arg == "-v")
            videoPath = argv[++i];
    }

    #pragma endregion

    #pragma region Calibration

    CalibrationData calibrationData;

    if (!calibrationData.LoadFromFile("Resources\\SaveData"))
    {
        vector<Mat> calibrationImages;

        for (const auto& file : directory_iterator(calibrationDirectory))
            calibrationImages.push_back(imread(file.path().string()));

        calibrationData = Calibrate(calibrationImages, Size(9, 6), 1.0);

        calibrationData.OutputToFile("Resources\\SaveData");
    }

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