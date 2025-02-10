from ultralytics import YOLO
import cv2 as cv
import numpy as np

from infoCollect import detection

class transfer:
    def __init__(self, image_Path, x_c, y_c):
        self.image_Path = image_Path
        self.x_c = x_c
        self.y_c = y_c
        self.width = 1280
        self.height = 720
        self.widthOut = 1280
        self.heightOut = 720
        self.bboxes = [(0.261403, 0.520626, 0.155335, 0.119971), 
                       (0.102714, 0.560461, 0.204733, 0.25126), 
                       (0.517697, 0.471291, 0.0823609, 0.0306483), 
                       (0.482397, 0.472682, 0.08892244, 0.0451083), 
                       (0.438851, 0.46933, 0.00844402, 0.0433482), 
                       (0.52338, 0.470639, 0.0387951, 0.0313609), 
                       (0.588466, 0.473114, 0.0587534, 0.0588323), 
                       (0.536375, 0.469547, 0.0270128, 0.0299886), 
                       (0.347094, 0.483236, 0.157202, 0.0854932), 
                       (0.322932, 0.491035, 0.11132, 0.0959795)]
        
    # Resize the bounding box of the detected item
    def resize(self):
        # Generate the pixel coordinates of each detected items' bounding box
        for i in self.bboxes:
            # Convert the x_center, y_center into pixel ccordinates form
            x_centerPixel = i[0] * self.width
            y_centerPixel = i[1] * self.height
            # Convert the width and height into pixel form
            widthPixel = i[2] * self.width
            heightPixel = i[3] * self.height
            # Get the four corner points of each bounding box
            topLeft = (x_centerPixel - widthPixel, y_centerPixel + heightPixel)
            topRight = (x_centerPixel + widthPixel, y_centerPixel + heightPixel)
            botRight = (x_centerPixel + widthPixel, y_centerPixel - heightPixel)
            botLeft = (x_centerPixel - widthPixel, y_centerPixel - heightPixel)
        
        

    # Transform the selected ROI
    def transform(self, image):

        pt_A = [500, 353]
        pt_B = [797, 353]
        pt_C = [1091, 483]
        pt_D = [145, 483]

        width_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_DC = np.sqrt(((pt_D[0] - pt_C[0]) ** 2) + ((pt_D[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AB), int(width_DC))

        height_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        height_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] + pt_C[1]) ** 2))
        maxHeight = max(int(height_AD), int(height_BC))
        
        srcPts = np.float32([pt_A, pt_B, pt_C, pt_D])
        dtPts = np.float32([0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0])

        matrix = cv.getPerspectiveTransform(srcPts, dtPts)
        transformed_frame = cv.warpPerspective(image, matrix, (maxWidth, maxHeight), flags = cv.INTER_LINEAR)

        cv.imshow("transformed_frame", transformed_frame)
        

def main():
    # test Image path
    testImg_Path = r"/home/lintan/BEV_ws/BEV-Space/Front_Image/BEV_front_test1.png"
    image_show = detection("coco8.yaml", "yolo11n.yaml", "yolo11n.pt", testImg_Path)
    image_show.detect()
    k = cv.waitKey(0)

if __name__ == '__main__': 
    main()


        
