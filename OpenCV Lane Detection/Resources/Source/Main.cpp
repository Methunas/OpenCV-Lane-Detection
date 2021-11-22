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

void ProcessFrame(Mat& frame, CalibrationData& calibrationData, UndistortMapData& undistortMapData, bool showStepsInNewWindows, bool combineStepsInFinalFrame)
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

    if (showStepsInNewWindows)
        imshow("Sky View", skyView);

    #pragma endregion

    #pragma region Lane Filter

    LaneFilterArgs laneFilterArgs(220, 40, 205, Point2f(0.7f, 1.4f), 40, 20);

    LaneFilterData laneFilterData;
    LaneFilter(frame, laneFilterData, laneFilterArgs);
    SkyView(laneFilterData.combinedMask, laneFilterData.combinedMask, sourcePoints, destinationPoints);

    Mat binary;
    threshold(laneFilterData.combinedMask, binary, 150, 1, THRESH_BINARY);

    rectangle(laneFilterData.combinedMask, Point(0, 0), Point(125, laneFilterData.combinedMask.rows), Scalar(0, 0, 0), -1);
    rectangle(laneFilterData.combinedMask, Point(laneFilterData.combinedMask.cols, 0), Point(1200, laneFilterData.combinedMask.rows), Scalar(0, 0, 0), -1);

    if (showStepsInNewWindows)
    {
        imshow("Color Threshold", laneFilterData.colorMask);
        imshow("Sobel Threshold", laneFilterData.sobelMask);
        imshow("Lane Filter", binary * 255);
    }

    #pragma endregion

    #pragma region Curve Fitting

    CurveFitData curveData;
    double metersPerPixel = 3.7 / 700;
    CurveFit(binary, curveData, metersPerPixel, 9, 200, 50);

    if (showStepsInNewWindows)
        imshow("Curve Fitting", curveData.image);

    #pragma endregion

    #pragma region Lane Projection

    Mat projected;
    ProjectLane(frame, projected, destinationPoints, sourcePoints, curveData);

    if (showStepsInNewWindows)
        imshow("Lane Projection", projected);

    #pragma endregion

    frame = projected;

    string posText = "Vehicle Position: " +
        to_string(curveData.vehiclePosition) +
        (curveData.vehiclePosition < 0 ? "m left" : "m right") +
        " of center";
    
    if (combineStepsInFinalFrame)
    {
        double viewWidth = frame.cols / 5.0;
        double viewHeight = frame.rows / 5.0;

        resize(skyView, skyView, Size(), 0.2, 0.2);
        resize(laneFilterData.colorMask, laneFilterData.colorMask, Size(), 0.2, 0.2);
        resize(laneFilterData.sobelMask, laneFilterData.sobelMask, Size(), 0.2, 0.2);
        resize(binary, binary, Size(), 0.2, 0.2);
        resize(curveData.image, curveData.image, Size(), 0.2, 0.2);

        binary *= 255;

        cvtColor(laneFilterData.colorMask, laneFilterData.colorMask, COLOR_GRAY2BGR);
        cvtColor(laneFilterData.sobelMask, laneFilterData.sobelMask, COLOR_GRAY2BGR);
        cvtColor(binary, binary, COLOR_GRAY2BGR);
        cvtColor(curveData.image, curveData.image, COLOR_GRAY2BGR);

        skyView.copyTo(frame(Rect(0, 0, viewWidth, viewHeight)));
        laneFilterData.colorMask.copyTo(frame(Rect(viewWidth, 0, viewWidth, viewHeight)));
        laneFilterData.sobelMask.copyTo(frame(Rect(viewWidth * 2, 0, viewWidth, viewHeight)));
        binary.copyTo(frame(Rect(viewWidth * 3, 0, viewWidth, viewHeight)));
        curveData.image.copyTo(frame(Rect(viewWidth * 4, 0, viewWidth, viewHeight)));

        rectangle(frame, Rect(0, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth * 2, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth * 3, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth * 4, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));

        putText(frame, posText, Point(15, viewHeight + 20), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
        // radius info
    }
    else
    {
        putText(frame, posText, Point(15, 20), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
        // radius info
    }
}

int main(int argc, char* argv[])
{
    #pragma region Argument Parse

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    string calibrationDirectory;
    string videoPath;
    bool showStepsInNewWindows = false;
    bool combineStepsInFinalFrame = false;

    for (int i = 0; i < argc; i++)
    {
        string arg(argv[i]);

        if (arg == "-c")
            calibrationDirectory = argv[++i];
        if (arg == "-v")
            videoPath = argv[++i];
        if (arg == "-n")
            showStepsInNewWindows = true;
        if (arg == "-m")
            combineStepsInFinalFrame = true;
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
        
        ProcessFrame(frame, calibrationData, undistortMapData, showStepsInNewWindows, combineStepsInFinalFrame);

        imshow("Lane Detection", frame);

        waitKey(frameDelay);
    }

    waitKey(0);
    return 0;
}