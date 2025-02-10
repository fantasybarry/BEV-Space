from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch

class detection:

    def __init__(self, trainData, model, preTrained, img_Path):
        self.epochs = 50
        self.imgsz = 1280
        self.trainData = trainData
        self.model = model
        self.preTrained = preTrained
        self.img_Path = img_Path

    def detect(self):
        # Load the YOLO11 Model
        detection_Model = YOLO(self.model)
        # Load a pretrained YOLO model
        detection_Model = YOLO(self.preTrained)
        # Train the model on the COCO8 dataset for 100 epochs
        results = detection_Model.train(data = self.trainData, epochs = self.epochs, imgsz = self.imgsz)
        # Run the inference with the YOLO11n model on the test image
        results = detection_Model(self.img_Path, save = False, show = True, save_txt = False)

        
    # draw the object tracking IDs
    #def draw_ID(self):
    # TODO

# depth Estimation class
#class Estimation:
    
# TODO



def main():
    # test Image path
    testImg_Path = r"/home/lintan/BEV_ws/BEV-Space/Front_Image/BEV_front_test1.png"
    image_show = detection("coco8.yaml", "yolo11n.yaml", "yolo11n.pt", testImg_Path)
    image_show.detect()
    k = cv.waitKey(0)

if __name__ == '__main__': 
    main()
    