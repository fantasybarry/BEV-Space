import cv2 as cv
import numpy as np
import torch
import os
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, Pad
from PIL import Image
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Initialize models once (outside video loop)
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' 
dataset = 'vkitti'
max_depth = 80

# Depth model
depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()

# Constants
BEV_WIDTH = 720
BEV_HEIGHT = 1280
M_PER_PIXEL_X = 50 / BEV_WIDTH
M_PER_PIXEL_Y = 40 / BEV_HEIGHT
IMAGE_WIDTH = 1920  # Must match target_width
FOV_HORIZONTAL = 76 # Need to be adjust
FOCAL_LENGTH_PX = (IMAGE_WIDTH / 2) / np.tan(np.radians(FOV_HORIZONTAL / 2))
TARGET_HEIGHT = 1080
TARGET_WIDTH = 1920

def detection(frame):
    """Using YOLO v11 for object detection"""

    #frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Load the yolo model
    yolo_model = YOLO('yolo11n.pt')
    results = yolo_model(source=frame, conf=0.7, imgsz = 1920, show = False, save = False) 
    return results

def convertFramesToVideo(path, out_path, out_video_name):
    """Convert a sequence of frames into video"""

    out_video_full_path = out_path + out_video_name
    pre_imgs = os.listdir(path)
    img = []

    for i in pre_imgs:
        i = path + i
        img.append(i)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame = cv.imread(img[0])
    
    size = list(frame.shape)
    del size[2]
    size.reverse()
    video = cv.VideoWriter(out_video_full_path, fourcc, 24, size)
    
    for i in range(len(img)):
        video.write(cv.imread(img[i]))
        print('frame', i+1, 'of', len(img))

    video.release()
    print('Generated Video to ', out_path)

def frameExtract(path):
    """Extract frames from the video"""

    cap = cv.VideoCapture(path)
    currentFrame = 0
    if cap.isOpened():

        while True:
            success, frame = cap.read()
            if success:
                cv.imwrite(r'./extractFrames/frame' + str(currentFrame) + '.jpg', frame)
                currentFrame += 1
            else:
                break

        cap.release()       
    cv.destroyAllWindows()

def get_padding(size, patch_size = 14):
    """Process the frame"""

    remainder = size % patch_size
    return 0 if remainder == 0 else patch_size - remainder
 
def depth(target_height, target_width, pad_width, pad_height, frame, yoloResults,currentFrame):
    """Get the distance for each objects of each frame"""

    pil_image = Image.fromarray(frame)

    transform = Compose([
    Resize((target_height, target_width)),Pad(padding=(pad_width//2, pad_height//2, pad_width//2, pad_height//2), 
                                              padding_mode = "constant"),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE) # Add batch dimension
    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        raise ValueError("Input tensor contains NaN/Inf values!")

    with torch.no_grad():
        depth = depth_model(input_tensor).squeeze().cpu().numpy() # Shape: (1, H, W)

    # depth normalized
    depth = depth[pad_height//2:-pad_height//2, pad_width//2:-pad_width//2]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 15

    boxesXYXY = yoloResults[0].boxes.xyxy
    avg_depths = []
    currentFrame = 0
    # Get distance for each object
    for box in boxesXYXY:
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        
        cropped_depth = depth_normalized[y1:y2, x1:x2]
        valid_depths = cropped_depth[(cropped_depth >= 0) & ~np.isnan(cropped_depth)]
        avg_depth = np.mean(valid_depths) if valid_depths.size > 0 else -1
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        avg_depths.append(avg_depth)
        
    cv.imwrite(r'./extractFramesYolo/frame' + str(currentFrame) + '.jpg', frame)
    
    return avg_depths

def bev(yoloResults, avg_depths, currentFrame):
    """Generate BEV Space frame"""

    bev_image = np.zeros((1280, 720, 3), dtype=np.uint8)
    ego_position = (360, 1270)
    cv.circle(bev_image, ego_position, 10, (0, 0, 255), -1)

    boxesXYWH = yoloResults[0].boxes.xywh.cpu().numpy()
    
    for i, box in enumerate(boxesXYWH):
        x_center, y_center, _, _ = box.astype(int)
        avg_depth = avg_depths[i]

        pixel_offset = x_center - (IMAGE_WIDTH // 2)
        lateral_distance = (pixel_offset / FOCAL_LENGTH_PX) * avg_depth

        bev_x_m = avg_depth
        bev_y_m = lateral_distance

        bev_x_px = int(360 + bev_y_m / M_PER_PIXEL_X)
        bev_y_px = int(1270 - bev_x_m / M_PER_PIXEL_Y)

        bev_x_px = max(0, min(bev_x_px, BEV_WIDTH - 1))
        bev_y_px = max(0, min(bev_y_px, BEV_HEIGHT - 1))
        
        cv.circle(bev_image, (bev_x_px, bev_y_px), 5, (0, 255, 0), -1)
        cv.putText(bev_image, f"{avg_depth:.1f}m", (bev_x_px+20, bev_y_px+10),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    cv.imwrite(r'./extractFramesBEV/frame' + str(currentFrame) + '.jpg', bev_image)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Main Function
if __name__ == '__main__':
    
    testVideoPath = r"/home/lintan/Depth-Anything-V2/metric_depth/test_Videos/test_video_indoor.MOV"
    frameExtract(path=testVideoPath)
    #detection(testVideoPath)
    
    extractFramesPath = r"/home/lintan/Depth-Anything-V2/metric_depth/extractFrames/" 
   # generatedVideoPath = r"/home/lintan/Depth-Anything-V2/metric_depth/extractFramesBEV/Video/" 
   # generatedVideoName = r'test_1.mp4'

    pad_height = get_padding(TARGET_HEIGHT)
    pad_width = get_padding(TARGET_WIDTH)

    extractFrames = load_images_from_folder(extractFramesPath)
    currentFrame = 0
    for extractFrame in extractFrames:
        yoloResults = detection(extractFrame)
        avgDepths = depth(TARGET_HEIGHT, TARGET_WIDTH, pad_width, pad_height, extractFrame, yoloResults, currentFrame)
        bev(yoloResults, avgDepths, currentFrame)
        currentFrame += 1
    
   # convertFramesToVideo(extractFramesPath, generatedVideoPath, generatedVideoName)
    