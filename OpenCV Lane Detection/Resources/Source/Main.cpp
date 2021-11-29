#include "Calibration.h"
#include "LaneFilter.h"
#include "SkyView.h"
#include "Curves.h"

#include <iostream>
#include <filesystem>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace filesystem;

struct PartUndistortMapData
{
    vector<Mat> map1_parts, map2_parts;
};

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

void ProcessFrame(Mat& frame, const CalibrationData& calibrationData, const PartUndistortMapData& undistortMapData, bool showStepsInNewWindows, bool combineStepsInFinalFrame)
{
    TickMeter timer;
    timer.start();

    // Undistort the frame using the calibration data
    //undistort(frame.clone(), frame, calibrationData.camMatrix, calibrationData.distortion);
    Mat undistorted;
    RemapFrame(frame, undistorted, calibrationData, undistortMapData);
    frame = undistorted;
    // Below function is meant to be used as a faster replacement to undistort, but it doesn't have the same output
    //remap(frame.clone(), frame, undistortMapData.map1, undistortMapData.map2, INTER_LINEAR, BORDER_CONSTANT);
    
    timer.stop();
    cout << "Undistort Time: " << timer.getTimeSec() << "s" << endl;

    timer.reset();
    timer.start();

    // Define the points which will be used to warp the perspective
    // Points from the car towards the horizon, aligned with a centered lane
    vector<Point2f> sourcePoints({ {580, 460}, {205, 720}, {1110, 720}, {703, 460} });
    vector<Point2f> destinationPoints({ {320, 0}, {320, 720}, {960, 720}, {960, 0} });

    Mat skyView;
    SkyView(frame, skyView, sourcePoints, destinationPoints);

    if (showStepsInNewWindows)
        imshow("Sky View", skyView);

    timer.stop();
    cout << "Sky View Time: " << timer.getTimeSec() << "s" << endl;

    timer.reset();
    timer.start();

    // Filter out the lane using a color mask and sobel mask on the saturation and lightness of the image
    LaneFilterArgs laneFilterArgs(220, 40, 205, Point2f(0.7f, 1.4f), 40, 20);

    LaneFilterData laneFilterData;
    LaneFilter(frame, laneFilterData, laneFilterArgs);
    SkyView(laneFilterData.combinedMask, laneFilterData.combinedMask, sourcePoints, destinationPoints);

    Mat binary;
    threshold(laneFilterData.combinedMask, binary, 150, 1, THRESH_BINARY);

    // Remove unnecessary edge points
    rectangle(laneFilterData.combinedMask, Point(0, 0), Point(125, laneFilterData.combinedMask.rows), Scalar(0, 0, 0), -1);
    rectangle(laneFilterData.combinedMask, Point(laneFilterData.combinedMask.cols, 0), Point(1200, laneFilterData.combinedMask.rows), Scalar(0, 0, 0), -1);

    if (showStepsInNewWindows)
    {
        imshow("Color Threshold", laneFilterData.colorMask);
        imshow("Sobel Threshold", laneFilterData.sobelMask);
        imshow("Lane Filter", binary * 255);
    }

    timer.stop();
    cout << "Lane Filter Time: " << timer.getTimeSec() << "s" << endl;

    timer.reset();
    timer.start();

    // Fit a curve to the lane points from the lane filtering
    CurveFitData curveData;
    double metersPerPixelX = 3.7 / 700;
    double metersPerPixelY = 30.0 / 720;

    CurveFit(binary, curveData, metersPerPixelX, metersPerPixelY, 9, 200, 10);

    if (showStepsInNewWindows)
        imshow("Curve Fitting", curveData.image);

    timer.stop();
    cout << "Curve Fit Time: " << timer.getTimeSec() << "s" << endl;

    timer.reset();
    timer.start();

    // Fill in the lane pixels and undo the sky view perspective warp
    Mat projected;
    ProjectLane(frame, projected, destinationPoints, sourcePoints, curveData);

    if (showStepsInNewWindows)
        imshow("Lane Projection", projected);

    frame = projected;

    timer.stop();
    cout << "Project Time: " << timer.getTimeSec() << "s" << endl;

    timer.reset();
    timer.start();

    // Add the results to the frame
    string posText = "Vehicle Position: " +
        to_string(abs(curveData.vehiclePosition)) +
        (curveData.vehiclePosition < 0 ? "m right" : "m left") +
        " of center";

    string leftRadiusText = "Left Radius: " + to_string(curveData.leftRadius);
    string rightRadiusText = "Right Radius: " + to_string(curveData.rightRadius);
    
    if (combineStepsInFinalFrame)
    {
        // Resize the individual frames to 1/5 their size
        double viewWidth = frame.cols / 5.0;
        double viewHeight = frame.rows / 5.0;

        resize(skyView, skyView, Size(), 0.2, 0.2);
        resize(laneFilterData.colorMask, laneFilterData.colorMask, Size(), 0.2, 0.2);
        resize(laneFilterData.sobelMask, laneFilterData.sobelMask, Size(), 0.2, 0.2);
        resize(binary, binary, Size(), 0.2, 0.2);
        resize(curveData.image, curveData.image, Size(), 0.2, 0.2);

        binary *= 255; // Needs to be multiplied to be visible

        // Convert from CV_8U to CV_8UC3
        cvtColor(laneFilterData.colorMask, laneFilterData.colorMask, COLOR_GRAY2BGR);
        cvtColor(laneFilterData.sobelMask, laneFilterData.sobelMask, COLOR_GRAY2BGR);
        cvtColor(binary, binary, COLOR_GRAY2BGR);

        // Copy the frames to the top of the final frame
        skyView.copyTo(frame(Rect(0, 0, viewWidth, viewHeight)));
        laneFilterData.colorMask.copyTo(frame(Rect(viewWidth, 0, viewWidth, viewHeight)));
        laneFilterData.sobelMask.copyTo(frame(Rect(viewWidth * 2, 0, viewWidth, viewHeight)));
        binary.copyTo(frame(Rect(viewWidth * 3, 0, viewWidth, viewHeight)));
        curveData.image.copyTo(frame(Rect(viewWidth * 4, 0, viewWidth, viewHeight)));

        // Draw rectangles around the frames to show their borders
        rectangle(frame, Rect(0, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth * 2, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth * 3, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));
        rectangle(frame, Rect(viewWidth * 4, 0, viewWidth, viewHeight), Scalar_(0, 0, 255));

        // Draw the lane data as text
        putText(frame, posText, Point(15, viewHeight + 20), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
        putText(frame, leftRadiusText, Point(frame.cols - 400, viewHeight + 20), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
        putText(frame, rightRadiusText, Point(frame.cols - 400, viewHeight + 40), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
    }
    else
    {
        // Draw the lane data as text
        putText(frame, posText, Point(15, 20), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
        putText(frame, leftRadiusText, Point(frame.cols - 400, 20), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
        putText(frame, rightRadiusText, Point(frame.cols - 400, 40), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, FILLED);
    }

    timer.stop();
    cout << "Combine Time: " << timer.getTimeSec() << "s" << endl << endl;
}

int main(int argc, char* argv[])
{
    // Parse the arguments
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

    // Calibrate the camera with all of the images in the SaveData folder
    // Load from file if it has already been calibrated
    CalibrationData calibrationData;

    if (!calibrationData.LoadFromFile("Resources\\SaveData"))
    {
        vector<Mat> calibrationImages;

        for (const auto& file : directory_iterator(calibrationDirectory))
            calibrationImages.push_back(imread(file.path().string()));

        calibrationData = Calibrate(calibrationImages, Size(9, 6), 1.0);

        calibrationData.OutputToFile("Resources\\SaveData");
    }

    // Read the video from the specified path and get its file properties
    VideoCapture video(videoPath);
    assert(video.isOpened());

    double fps = video.get(VideoCaptureProperties::CAP_PROP_FPS);
    int frameDelay = (int)(1000 / fps);
    Size videoSize((int)video.get(cv::CAP_PROP_FRAME_WIDTH), (int)video.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Generate the undistort maps for use with the RemapFrame() function in ProcessFrame()
    PartUndistortMapData partUndistortMapData;
    CalculatePartUndistortMaps(partUndistortMapData, videoSize, calibrationData);

    for (;;)
    {
        Mat frame;
        video >> frame;

        // Reset the frame position if all frames have been displayed
        if (frame.empty())
        {
            video.set(VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        
        ProcessFrame(frame, calibrationData, partUndistortMapData, showStepsInNewWindows, combineStepsInFinalFrame);

        imshow("Lane Detection", frame);
        waitKey(frameDelay);
    }

    waitKey(0);
    return 0;
}