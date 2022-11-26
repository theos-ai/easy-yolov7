# 🤙🏻 Easy YOLOv7 ⚡️

![Easy YOLOv7 by Theos AI](detected_image.jpg)

This a clean and easy-to-use implementation of [YOLOv7](https://github.com/WongKinYiu/yolov7) in PyTorch, made with ❤️ by [Theos AI](https://theos.ai).

Don't forget to read our [Blog](https://blog.theos.ai) and subscribe to our [YouTube Channel](https://www.youtube.com/@theos-ai/)!

### Install all the dependencies

```
pip install -r requirements.txt
```

### Detect the image

```
python image.py
```

### Detect the webcam

```
python webcam.py
```

### Detect the video

```
python video.py
```

https://user-images.githubusercontent.com/14842535/204094120-8fc55f91-cc30-4097-9ad5-06f3cbc27b9c.mp4

## Train on your own custom dataset

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/MorMkGS6_WU/0.jpg)](https://www.youtube.com/watch?v=MorMkGS6_WU)

### Click the weights button

![Download weights button of Theos AI](assets/button.jpg)

### Download the files
Download the best or last weights and the classes YAML file and put them inside this directory.

![Download weights modal of Theos AI](assets/weights.jpg)

### Change this line to use your custom model

``` Python
yolov7.load('best.weights', classes='classes.yaml', device='cpu') # use 'gpu' for CUDA GPU inference
```