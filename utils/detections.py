from collections import OrderedDict
from PIL import ImageColor
import numpy as np
import json
import cv2


class Point:
    def __init__(self, raw_point):
        self.x = raw_point[0]
        self.y = raw_point[1]

    def to_string(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'
    
    def to_dict(self):
        return {'x':self.x, 'y':self.y}


class Box:
    def __init__(self, class_name, confidence, raw_corner_points, color, track_id=None):
        self.class_name = class_name
        self.confidence = confidence
        self.raw_corner_points = raw_corner_points
        self.top_left_point = Point(raw_corner_points[0])
        self.bottom_right_point = Point(raw_corner_points[1])
        self.width =  self.bottom_right_point.x - self.top_left_point.x
        self.height = self.bottom_right_point.y - self.top_left_point.y
        self.color = color
        self.track_id = track_id

    def to_dict(self):
        box = OrderedDict([
            ('class', self.class_name),
            ('confidence', self.confidence),
            ('x', self.top_left_point.x),
            ('y', self.top_left_point.y),
            ('width', self.width),
            ('height', self.height),
            ('color', self.color)
        ])
        if self.track_id is not None:
            box['id'] = self.track_id
        return box


class Detections:
    def __init__(self, raw_detection, classes, tracking=False):
        self.__raw_detection = raw_detection
        self.__classes = classes
        self.__boxes = []
        self.__tracking = tracking
        self.__point1_index = 0
        self.__point2_index = 1
        self.__point3_index = 2
        self.__point4_index = 3
        self.__tracking_index = 4
        self.__class_index = 5 if tracking else 5
        self.__confidence_index = 6 if tracking else 4
        self.__extract_boxes()

    def __extract_boxes(self):
         for raw_box in self.__raw_detection:
            track_id = None
            if self.__tracking:
                track_id = int(raw_box[self.__tracking_index])
            class_id = int(raw_box[self.__class_index])
            raw_corner_points = (int(raw_box[self.__point1_index]), int(raw_box[self.__point2_index])), (int(raw_box[self.__point3_index]), int(raw_box[self.__point4_index]))
            confidence = raw_box[self.__confidence_index]
            dataset_class = self.__classes[class_id]
            class_name = dataset_class['name']
            class_color = dataset_class['color']
            box = Box(class_name, confidence, raw_corner_points, class_color, track_id=track_id)
            self.__boxes.append(box)
        
    def get_boxes(self):
        return self.__boxes

    def to_dict(self):
        boxes = []
        for box in self.__boxes:
            boxes.append(box.to_dict())
        return boxes

    def to_json(self):
        boxes = self.to_dict()
        return json.dumps(boxes, indent=4)


def plot_box(image, top_left_point, bottom_right_point, width, height, label, color=(210,240,0), padding=6, font_scale=0.35):
    label = label.upper()
    
    cv2.rectangle(image, (top_left_point['x'] - 1, top_left_point['y']), (bottom_right_point['x'], bottom_right_point['y']), color, thickness=2, lineType=cv2.LINE_AA)
    res_scale = (image.shape[0] + image.shape[1])/1600
    font_scale = font_scale * res_scale
    font_width, font_height = 0, 0
    font_face = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(label, font_face, fontScale=font_scale, thickness=1)[0]

    if text_size[0] > font_width:
        font_width = text_size[0]
    if text_size[1] > font_height:
        font_height = text_size[1]
    if top_left_point['x'] - 1 < 0:
        top_left_point['x'] = 1
    if top_left_point['x'] + font_width + padding*2 > image.shape[1]:
        top_left_point['x'] = image.shape[1] - font_width - padding*2
    if top_left_point['y'] - font_height - padding*2  < 0:
        top_left_point['y'] = font_height + padding*2
    
    p3 = top_left_point['x'] + font_width + padding*2, top_left_point['y'] - font_height - padding*2
    cv2.rectangle(image, (top_left_point['x'] - 2, top_left_point['y']), p3, color, -1, lineType=cv2.LINE_AA)
    x = top_left_point['x'] + padding
    y = top_left_point['y'] - padding
    cv2.putText(image, label, (x, y), font_face, font_scale, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
    return image

def draw(image, detections):
    image_copy = image.copy()
    for box in detections:
        class_name = box['class']
        conf = box['confidence']
        text = ''
        if 'text' in box:
            text = box['text']
            if len(text) > 50:
                text = text[:50] + ' ...'
        label = (str(box['id']) + '. ' if 'id' in box else '') + class_name + ' ' + str(int(conf*100)) + '%' + ((' | ' + text) if ('text' in box and len(box['text']) > 0 and not box['text'].isspace()) else '')
        width = box['width']
        height = box['height']
        color = box['color']

        if isinstance(color, str):
            color = ImageColor.getrgb(color)
            color = (color[2], color[1], color[0])
        
        top_left_point = {'x':box['x'], 'y':box['y']}
        bottom_right_point = {'x':box['x'] + width, 'y':box['y'] + height}
        image_copy = plot_box(image_copy, top_left_point, bottom_right_point, width, height, label, color=color)
    return image_copy