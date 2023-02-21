from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
import cv2

yolov7 = YOLOv7()
ocr_classes=['license-plate']
yolov7.set(ocr_classes=ocr_classes)
yolov7.load('anpr.weights', classes='anpr.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

video = cv2.VideoCapture('vehicles.mp4')
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

if video.isOpened() == False:
	print('[!] error opening the video')

print('[+] started reading text on the video...\n')
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
texts = {}

try:
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            detections = yolov7.detect(frame, track=True)
            
            for detection in detections:
                if detection['class'] in ocr_classes:
                    detection_id = detection['id']
                    text = detection['text']
                    if len(text) > 0:
                        if detection_id not in texts:
                            texts[detection_id] = {
                                'most_frequent':{
                                    'value':'', 
                                    'count':0
                                }, 
                                'all':{}
                            }
                        
                        if text not in texts[detection_id]['all']:
                            texts[detection_id]['all'][text] = 0
                        
                        texts[detection_id]['all'][text] += 1

                        if texts[detection_id]['all'][text] > texts[detection_id]['most_frequent']['count']:
                            texts[detection_id]['most_frequent']['value'] = text
                            texts[detection_id]['most_frequent']['count'] = texts[detection_id]['all'][text]
                
                    if detection_id in texts:
                        detection['text'] = texts[detection_id]['most_frequent']['value']

            detected_frame = draw(frame, detections)
            output.write(detected_frame)
            pbar.update(1)
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
output.release()
yolov7.unload()