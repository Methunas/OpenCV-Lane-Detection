#include "Calibration.h"
#include "FrameProcessing.h"

#include <filesystem>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace filesystem;

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