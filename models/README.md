# Detector models (not committed)

YOLOX by Megvii, Apache-2.0 licence: https://github.com/Megvii-BaseDetection/YOLOX
Chosen instead of Ultralytics YOLO (AGPL-3.0) so the project stays permissively licensed.

Download (release 0.1.1rc0):

    curl -L -o models/yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
    curl -L -o models/yolox_tiny.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx

SHA-256 of the files used during development:

    yolox_nano.onnx  c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
    yolox_tiny.onnx  427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7

Both take a 416x416 input and output COCO classes; only car, motorcycle, bus and truck are kept.
