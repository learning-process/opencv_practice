#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;

double confThreshold, nmsThreshold;

void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB)
{
    static Mat blob;
    blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);
    net.setInput(blob, "", scale, mean);
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

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

    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != DNN_BACKEND_OPENCV))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
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

    int backend = DNN_BACKEND_OPENCV;

    Net net = readNet(caffemodel_path, prototxt_path);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(DNN_TARGET_CPU);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    // ssd300 parameters
    double scale = 1.0;
    int inpWidth = 300;
    int inpHeight = 300;
    Scalar mean = {104.0, 117.0, 123.0};
    bool swapRB = false;
    bool crop = false;

    VideoCapture cap;
    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        std::cout << 123 ;

        preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);

        std::vector<Mat> outs;
        net.forward(outs, outNames);

        postprocess(frame, outs, net, backend);

        imshow("face-detection", frame);
    }

    return 0;
}