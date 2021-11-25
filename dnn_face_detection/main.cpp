#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend)
{

    double confThreshold = 0.2;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    CV_Assert(outs.size() > 0);
    for (size_t k = 0; k < outs.size(); k++)
    {
        float* data = (float*)outs[k].data;
        for (size_t i = 0; i < outs[k].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left   = (int)data[i + 3];
                int top    = (int)data[i + 4];
                int right  = (int)data[i + 5];
                int bottom = (int)data[i + 6];
                int width  = right - left + 1;
                int height = bottom - top + 1;
                if (width <= 2 || height <= 2)
                {
                    left   = (int)(data[i + 3] * frame.cols);
                    top    = (int)(data[i + 4] * frame.rows);
                    right  = (int)(data[i + 5] * frame.cols);
                    bottom = (int)(data[i + 6] * frame.rows);
                    width  = right - left + 1;
                    height = bottom - top + 1;
                }
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }

    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 255, 0));
    }
}

int main(int argc, char **argv) 
{
    std::string prototxt_path = argv[1];
    std::string caffemodel_path = argv[2];
    std::string image_path = argv[3];

    int backend = DNN_BACKEND_OPENCV;

    Net net = readNet(caffemodel_path, prototxt_path);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // ssd300 parameters
    Mat image = imread(image_path);
    Mat blob;
    double scale = 1.0;
    int inpWidth = 300;
    int inpHeight = 300;
    Scalar mean = {104.0, 117.0, 123.0};
    bool swapRB = false;
    bool crop = false;

    blobFromImage(image, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, crop);
    net.setInput(blob);

    std::vector<Mat> outs;
    net.forward(outs);

    while (waitKey(1) < 0)
    {
        postprocess(image, outs, net, backend);
        imshow("face-detection", image);
    }

    return 0;
}
