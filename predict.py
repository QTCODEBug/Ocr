
import cv2
import numpy as np
import matplotlib.pyplot as plt   
import tensorflow as tf

def predict_num(model,img):
  image=np.array([img])
  res=model.predict(image)
  index = np.argmax(res)
  return str(index)

startInference = False

def ifclicked(event, x , y, flags, params):
  global startInference
  if event==cv2.EVENT_LBUTTONDOWN:
    startInference= not startInference

threshold= 100
def on_threshould(x):
  global threshold
  threshold=x
def start_cv(model):
  global threshold
  cap=cv2.VideoCapture(0)
  frame=cv2.namedWindow('background')
  cv2.setMouseCallback('background',ifclicked)
  cv2.createTrackbar('threshold','background',150,255,on_threshould)
  background=np.zeros((480,640),np.uint8)
  frameCount=0

  while True:
    ret,frame = cap.read()
    if(startInference):
      frameCount +=1  
      frame[0:480,0:80 ] = 0
      frame[0:480,560:640] = 0
      grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      _, thr= cv2.threshold(grayframe,threshold,255,cv2.THRESH_BINARY_INV)
      resizeFrame=thr[240- 75:240 +75 , 320-75:320+75]
      background[240 - 75:240 +75 , 320 - 75:320 +75] = resizeFrame

      iconImg= cv2.resize(resizeFrame, (28,28))

      res= predict_num(model,iconImg)

      if frameCount==5:
        background[0:480,0:80] =0
        frameCount=0
      cv2.putText(background, res , (10,60), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255), 3)
      cv2.rectangle(background, (320 - 80,240 - 80), (320 + 80,240 + 80 ),(255,255,255), thickness=3)

      cv2.imshow('background',background)
    else:
      cv2.imshow('background',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

model=tf.keras.models.load_model('D:\OCR model\epic.model')
start_cv(model)