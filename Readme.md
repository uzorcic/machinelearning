jupyter notebook: Training a yolov8 model with coco set and custom settings

Other notes:
Use rectlabel app for macos for annotation of objects.

Training of yolo dataset with custom images (must be previously annotated):
yolo task=detect mode=train model=yolov8s.pt data=<path_to>/data.yaml epochs=100 imgsz=1280

imgsz must match the training images resolution

detection:
see detect.py

