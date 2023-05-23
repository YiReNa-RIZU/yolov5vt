import torch
from PIL import Image
import cv2
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img1 = Image.open('renne.jpg')  # PIL image
# img2 = cv2.imread('img_t/bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
# imgs = [img1, img2]  # batch of images

# Inference
results = model(img1, size=640)  # includes NMS

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
print(results.xyxy[0])  # img1 predictions (tensor)
print('----------------')
print(results.pandas().xyxy[0])  # img1 predictions (pandas)

results.save()