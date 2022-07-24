import importlib
from tkinter import Frame
from turtle import delay, width
from cv2 import KeyPoint
from py import process
from sympy import false, true
import tensorflow as tf ;
import numpy as np;
from matplotlib import pyplot as plt;
import cv2
from torch import dtype;
import threading

from test import Videocap

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite') # โหลดโมเดลจาก....
interpreter.allocate_tensors() # เพิ่มประสิทธิภาพการคาดการณ์

def draw_keypoint (frame , keypoint , confidance_t):
    y , x , c  = frame.shape
    shaped = np.squeeze(np.multiply(keypoint , [y,x,1]))

    for kp in shaped:
        ky , kx , kp_conf = kp
        if kp_conf > confidance_t:
            cv2.circle(frame,(int(kx),int(ky)),4,(0,255,0),-1)


# สร้างเส้นเชื่อม -----------------------------------------

def draw_connection(frame , keypoint , edges , confidance_t):
    y , x, c =frame.shape
    shaped = np.squeeze(np.multiply(keypoint , [y,x,1]))

    for edge , color in edges.items():
        p1,p2 = edge 
        y1 , x1 , c1 = shaped[p1]
        y2 , x2 , c2 = shaped[p2]

        if (c1 >confidance_t) & (c2>confidance_t):
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)) , (0,255,0) , 2)

#----------EDGES
EDGES = {
    (0,1):'m',
    (0,2):'c',
    #(1,3):'m',
    (2,4):'c',
    (0,5):'m',
    (0,6):'c',
    (5,7):'m',
    (7,9):'m',
    (6,8):'c',
    (8,10):'c',
    (5,6):'y',
    (5,11):'m',
    (6,12):'c',
    (11,12):'y',
    (11,13):'m',
    (13,15):'m',
    (12,14):'c',
    (14,16):'c'
}


def input_camera():
    score_camera = []
    cap = cv2.VideoCapture(0)
    try:
        while (cap.isOpened()):
            ret , frame = cap.read() 
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img , axis=0),192,192)
            input_image = tf.cast(img, dtype=tf.uint8)

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'],np.array(input_image))
            interpreter.invoke()
            KeyPoint_with_scores = interpreter.get_tensor(output_details[0]['index'])
            score_camera.append(KeyPoint_with_scores)

            draw_connection(frame , KeyPoint_with_scores , EDGES  , 0.2)
            draw_keypoint(frame , KeyPoint_with_scores , 0.2)

            cv2.imshow('test' , frame)

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        

    except:
        cap.release()
        cv2.destroyAllWindows()
        print("type error")
    return cap


def input_Video():
    score_model = []
    cap = cv2.VideoCapture('image/1.avi')
    try:
        while (cap.isOpened()):
            ret , frame = cap.read() 
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img , axis=0),192,192)
            input_image = tf.cast(img, dtype=tf.uint8)

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'],np.array(input_image))
            interpreter.invoke()
            KeyPoint_with_scores = interpreter.get_tensor(output_details[0]['index'])
            score_model.append(KeyPoint_with_scores)

            draw_connection(frame , KeyPoint_with_scores , EDGES  , 0.2)
            draw_keypoint(frame , KeyPoint_with_scores , 0.2)

            cv2.imshow('test' , frame)

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        

    except:
        cap.release()
        cv2.destroyAllWindows()
        print("type error")
    return cap
    
if __name__ == "__main__":
    t1 = threading.Thread(target=input_camera , args= "" )
    t2 = threading.Thread(target=input_Video , args= "" )

    t1.start()
    t2.start()

    t1.join()
    t2.join()