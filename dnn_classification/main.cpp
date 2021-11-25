#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;

int main(int argc, char **argv) 
{
    std::string prototxt_path = argv[1];
    std::string caffemodel_path = argv[2];
    std::string image_path = argv[3];

    Net net = readNet(caffemodel_path, prototxt_path);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat image = imread(image_path);
    Mat blob;
    // Densnet-121 parameters
    double scale = 1 / 58.8235294117647;
    int inpWidth = 224;
    int inpHeight = 224;
    Scalar mean = {103.94, 116.78, 123.68};
    bool swapRB = false;
    bool crop = false;

    blobFromImage(image, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, crop);

    net.setInput(blob);

    Mat prob = net.forward();

    double confidence;
    Point classIdPoint;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    std::cout << "classId = " << classId << std::endl;

    return 0;
}