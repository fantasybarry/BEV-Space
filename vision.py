import cv2 as cv
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Create a new YOLO model from scratch
#model = YOLO("yolo11n.yaml")



class vision:
    

    def __init__(self, image_path, src_points, bev_x_range, bev_z_range, pixels_per_meter=10):
        
        # Read the image from the path
        self.image = cv.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        # Convert the image to RGB
        self.image_rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

        # Define the source points for the perspective transformation(Camera Space)
        self.src_points = np.array(src_points, dtype=np.float32)

        # setup the bird's eye view space
        self.x_min, self.x_max = bev_x_range
        self.z_min, self.z_max = bev_z_range
        self.pixels_per_meter = pixels_per_meter

        # compute the bev width and height
        self.bev_width = int((self.x_max - self.x_min) * self.pixels_per_meter)
        self.bev_height = int((self.z_max - self.z_min) * self.pixels_per_meter)

        # Define the destination points for the perspective transformation(BEV Space)
        self.dst_points = np.float32([[0, 0], [self.bev_width, 0], [self.bev_width, self.bev_height], [0, self.bev_height]])

        # Compute the homography
        self.homo_matrix = cv.getPerspectiveTransform(self.src_points, self.dst_points)

    # warpPerspective Transform Function
    def warp_transform(self):
        self.bev_image = cv.warpPerspective(self.image_rgb, self.homo_matrix,
                                            (self.bev_width, self.bev_height))
        return self.bev_image
    
    # Display function
    def display(self):
        # Ensure the bev is computed
        if not hasattr(self, 'bev_iamge'):
            self.warp_transform
        
        # create the copy of original image and overlay the source points
        image_with_points = self.image_rgb.copy()
        cv.polylines(image_with_points, [self.src_points.astype(np.int32)], True,
                     (255, 0, 0), 2)
        
        # Plot both images
        fig, axs = plt.subplots(1, 2, figsize = (12,6))

        axs[0].imshow(image_with_points)
        axs[0].set_title("Front Camera Image (Selected Region)")
        axs[0].axis("off")
        
        axs[1].imshow(self.bev_image)
        axs[1].set_title("BEV Image (x: right, z: forward)")
        axs[1].axis("off")
        
        plt.tight_layout()
        plt.show()
    

if __name__ == '__main__':

    image_path = r'/home/lintan/BEV_ws/BEV-Space/Front_Image/BEV_front_test1.png'
    src_points = [[400, 300], [800, 300], [850, 600], [350, 600]]
    bev_x_range = (-5, 5)
    bev_z_range = (0, 20)

    vision_system = vision(image_path, src_points, bev_x_range, bev_z_range, pixels_per_meter=10)
    # Obtain the bev image
    bev_image = vision_system.warp_transform()
    # Display
    vision_system.display()