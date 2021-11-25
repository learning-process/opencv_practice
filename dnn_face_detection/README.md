# OpenCV DNN Face Detection

## 1. Install OpenVINO python (get omz_downloader)
```
python3 -m pip install openvino-dev
```

## 2. Find model
[https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md)


## 3. Download model
```
omz_downloader --name ssd300
```

## 4. Build OpenCV 
```
cd opencv
mkdir build
cd build
cmake ..
cmake --build . 
cd ../..
```

## 5. Build dnn_face_detection 
```
cd dnn_face_detection
mkdir build
cd build
cmake -D OpenCV_DIR=../opencv/build/ ..
cmake --build . 
cd ../..
```

## 6. Run dnn_face_detection 
```
./dnn_face_detection/build/DNN_FACE_DETECTION public/ssd300/ssd300.prototxt public/ssd300/ssd300.caffemodel
```
