#include "FrameProcessing.h"

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
        int viewWidth = frame.cols / 5.0;
        int viewHeight = frame.rows / 5.0;

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