from paddleocr import PaddleOCR

paddle_ocr = None

def read(image):
  global paddle_ocr

  if paddle_ocr is None:
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
  
  result = paddle_ocr.ocr(image, cls=True)
  text = ''

  for idx in range(len(result)):
    res = result[idx]
    for line in res:
        if len(text) > 0:
            text += '; '
        text += f'{line[1][0]}'
  
  return {'text': text}