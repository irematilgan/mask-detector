
# import cv2
# import time
# import numpy as np
# from picamera.array import PiRGBArray
# import imutils
# from cv2 import CascadeClassifier
# CASCADE_PATH='/home/es123/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml'
# face_cascade = CascadeClassifier(CASCADE_PATH)

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
TRIG = 23
ECHO = 24

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

while True:
    GPIO.output(TRIG, False)
    time.sleep(2)
    GPIO.output(TRIG, True)
    time.sleep(0.0001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_interval = pulse_end - pulse_start
    interval = (pulse_interval/2)*34300
    interval = round(interval,2)

    if interval > 2 and interval < 400:
        print('Mesafe: ' + str(interval) + ' cm')