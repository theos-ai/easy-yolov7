from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2

yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference
image = cv2.imread('image.jpg')
detections = yolov7.detect(image)
detected_image = draw(image, detections)
cv2.imwrite('detected_image.jpg', detected_image)
print(json.dumps(detections, indent=4))