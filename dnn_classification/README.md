# OpenCV DNN Classification

## 1. Install OpenVINO python (get omz_downloader)
```
python3 -m pip install openvino-dev
```

## 2. Find model
[https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md)


## 3. Download model
```
omz_downloader --name densenet-121
```

## 4. Build OpenCV 
```
git clone https://github.com/opencv/opencv
cd opencv
mkdir build
cd build
cmake ..
cmake --build . 
cd ../..
```

## 5. Build dnn_classification 
```
cd dnn_classification
mkdir build
cd build
cmake -D OpenCV_DIR=../opencv/build/ ..
cmake --build . 
cd ../..
```

## 6. Run dnn_classification 
```
./dnn_classification/build/DNN_CLASSIFICATION public/densenet-121/densenet-121.prototxt public/densenet-121/densenet-121.caffemodel  dnn_classification/car.jpg
```
