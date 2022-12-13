
import cv2
import time
import numpy as np
from picamera.array import PiRGBArray
import imutils
from cv2 import CascadeClassifier
import RPi.GPIO as GPIO


CASCADE_PATH='/home/es123/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml'
COLOR_BLUE_MIN = np.array([118,0,75])
COLOR_BLUE_MAX = np.array([254,30,100])

COLOR_BLACK_MIN = np.array([0,0,0])
COLOR_BLACK_MAX = np.array([359,50,30])

COLOR_WHITE_MIN = np.array([0,0,75])
COLOR_WHITE_MAX = np.array([359,10,100])

TRIG = 23
ECHO = 24
BZ = 4
LEDY = 17
LEDG = 27

def nothing(x):
    pass

def detectMask(threshed):
    min_area = 15

    threshed = threshed/255
    total = np.sum(threshed)/(threshed.shape[0]*threshed.shape[1])
    maskOn = False
    if total >= min_area/100:
        maskOn = True
    
    return maskOn

def faceExist(frame):
    face_cascade = CascadeClassifier(CASCADE_PATH)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    faces_in_the_frame = 0
    for (x,y,w,h) in faces:
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        faces_in_the_frame +=1
    # print('Faces: ', str(faces_in_the_frame))
    # cv2.putText(frame, 'No. Faces in the frame:' + str(faces_in_the_frame), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 255), 1)
    print('No. Faces in the frame:' + str(faces_in_the_frame))
    return faces_in_the_frame >=1

def isClose(frame):
    GPIO.output(TRIG, False)
    time.sleep(0.00001)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    pulse_start = 0
    pulse_end = 0

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_interval = pulse_end - pulse_start
    interval = (pulse_interval/2)*34300
    interval = round(interval,2)

    if interval > 2 and interval < 400:
        # print(interval)
        # cv2.putText(frame, 'Distance from camera: ' + str(interval), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 255), 1)
        print('Distance from camera: ' + str(interval))

    return interval < 30.0
    

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)


    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.setup(BZ, GPIO.IN)
    GPIO.setup(BZ, GPIO.OUT)
    GPIO.setup(LEDY, GPIO.OUT)
    GPIO.setup(LEDG, GPIO.OUT)

    GPIO.output(LEDY, False)
    GPIO.output(LEDG, False)

    min_contour_area = 500
    resolution = (640, 480)
    fps = 32

    # camera = PiCamera()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,160.0)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,120.0)
    camera.set(cv2.CAP_PROP_FPS,30)
    # rawCapture = PiRGBArray(camera, size=resolution)
    ret, frame = camera.read()
    cv2.imshow("VideoFeed", frame)
    print("Starting video capture, press 'q' to exit")
    
    # @ $
    

    # H = 0-180, S = 0-255, V = 0-255
    # /360*180, /100*255, /100*255


    ## PANEL TO CONTROL HSV THRESHOLDS
    # cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    # cv2.createTrackbar('H_min', 'Controls', 0, 360, nothing)
    # cv2.createTrackbar('S_min', 'Controls', 0, 100, nothing)
    # cv2.createTrackbar('V_min', 'Controls', 0, 100, nothing)
    # cv2.setTrackbarPos('H_min', 'Controls', COLOR_WHITE_MIN[0])
    # cv2.setTrackbarPos('S_min', 'Controls', COLOR_WHITE_MIN[1])
    # cv2.setTrackbarPos('V_min', 'Controls', COLOR_WHITE_MIN[2])

    # cv2.createTrackbar('H_max', 'Controls', 0, 360, nothing)
    # cv2.createTrackbar('S_max', 'Controls', 0, 100, nothing)
    # cv2.createTrackbar('V_max', 'Controls', 0, 100, nothing)
    # cv2.setTrackbarPos('H_max', 'Controls', COLOR_WHITE_MAX[0])
    # cv2.setTrackbarPos('S_max', 'Controls', COLOR_WHITE_MAX[1])
    # cv2.setTrackbarPos('V_max', 'Controls', COLOR_WHITE_MAX[2])

    # cv2.createTrackbar('AreaSize', 'Controls', 0,100, nothing)
    # cv2.setTrackbarPos('AreaSize', 'Controls', 15)
    
    while True:
        try:
            ret, frame = camera.read()

            # frame = f.array
            # resize frame since processing on large raw image is costly/unnecessary
            # frame = imutils.resize(frame, width=500)
            frame = cv2.resize(frame, (160,120))
            isFace = faceExist(frame)
            close = isClose(frame)
            if not (isFace or close):
                # cv2.putText(frame, 'Mask On:' + 'Undefined', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
                print('Mask On:' + 'Undefined')
                GPIO.output(LEDY, False)
                GPIO.output(LEDG, False)
                GPIO.output(BZ, False)
                continue
            blurredFrame = cv2.GaussianBlur(frame, (25, 25), 0)
            # roi = blurredFrame[160:-190, 200:-200, :]
            roi = blurredFrame[40:-40, 50:-50, :]
            roiHSV = np.zeros(shape = roi.shape)
            roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # print(roiHSV[roiHSV>0])
            # H_min = cv2.getTrackbarPos('H_min', 'Controls')
            # S_min = cv2.getTrackbarPos('S_min', 'Controls')
            # V_min = cv2.getTrackbarPos('V_min', 'Controls')
            # H_max = cv2.getTrackbarPos('H_max', 'Controls')
            # S_max = cv2.getTrackbarPos('S_max', 'Controls')
            # V_max = cv2.getTrackbarPos('V_max', 'Controls')


            frame_threshed_BLUE = cv2.inRange(roiHSV, (COLOR_BLUE_MIN[0]/360*180,COLOR_BLUE_MIN[1]/100*255,COLOR_BLUE_MIN[2]/100*255), (COLOR_BLUE_MAX[0]/360*180,COLOR_BLUE_MAX[1]/100*255,COLOR_BLUE_MAX[2]/100*255))
            # frame_threshed_WHITE = cv2.inRange(roiHSV, (COLOR_WHITE_MIN[0]/360*180,COLOR_WHITE_MIN[1]/100*255,COLOR_WHITE_MIN[2]/100*255), (COLOR_WHITE_MAX[0]/360*180,COLOR_WHITE_MAX[1]/100*255,COLOR_WHITE_MAX[2]/100*255))
            frame_threshed_BLACK = cv2.inRange(roiHSV, (COLOR_BLACK_MIN[0]/360*180,COLOR_BLACK_MIN[1]/100*255,COLOR_BLACK_MIN[2]/100*255), (COLOR_BLACK_MAX[0]/360*180,COLOR_BLACK_MAX[1]/100*255,COLOR_BLACK_MAX[2]/100*255))
            # print(frame_threshed_BLUE[frame_threshed_BLUE>0])
            frame_threshed_BLUE = cv2.dilate(frame_threshed_BLUE, None, iterations=1)
            # frame_threshed_WHITE = cv2.dilate(frame_threshed_WHITE, None, iterations=1)
            frame_threshed_BLACK = cv2.dilate(frame_threshed_BLACK, None, iterations=1)
            # print(frame_threshed_BLUE[frame_threshed_BLUE>0])

            # frame_threshed = cv2.bitwise_or(frame_threshed_BLUE, frame_threshed_WHITE)
            # frame_threshed = cv2.bitwise_or(frame_threshed, frame_threshed_BLACK)
            contours_BLUE = cv2.findContours(frame_threshed_BLUE, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            # contours_WHITE = cv2.findContours(frame_threshed_WHITE.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
            contours_BLACK = cv2.findContours(frame_threshed_BLACK, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            # print(contours_BLUE)  
            # contours = cv2.findContours(frame_threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] 

            
            for (contour_BLUE, contour_BLACK) in zip(contours_BLUE, contours_BLACK):
                if cv2.contourArea(contour_BLUE) < min_contour_area:
                    continue

                (x1, y1, w1, h1) = cv2.boundingRect(contour_BLUE)
                # (x2, y2, w2, h2) = cv2.boundingRect(contour_WHITE)
                (x3, y3, w3, h3) = cv2.boundingRect(contour_BLACK)

                # print(x)
                # print(y)
                cv2.rectangle(frame, (120+x1 + 10,70+y1 + 10), (x1 + w1 + 120 + 10, y1 + 70 + h1 + 10), (0, 0, 255), 2)
                # cv2.rectangle(frame, (200+x2 + 10,160+y2 + 10), (x2 + w2 + 200 + 10, y2 + 160 + h2 + 10), (0, 0, 255), 2)
                cv2.rectangle(frame, (120+x3 + 10,70+y3 + 10), (x3 + w3 + 120 + 10, y3 + 70 + h3 + 10), (0, 0, 255), 2)
                # cv2.rectangle(frame_threshed, (x+10, y+10), (x + w, y + h), (255, 255, 255), 2)
        except Exception as e:
            print(e)
            continue

        # cv2.putText(frame, motionStatus, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        maskOn = False
        if detectMask(frame_threshed_BLUE) or detectMask(frame_threshed_BLACK):
            maskOn = True
            GPIO.output(BZ, False)
            GPIO.output(LEDG, True)
            GPIO.output(LEDY, False)
        else:
            GPIO.output(BZ, True)
            GPIO.output(LEDY, True)
            GPIO.output(LEDG, False)

        # cv2.putText(frame, 'Mask On:' + str(maskOn), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
        print('Mask On:' + str(maskOn))

        # cv2.imshow("Blurred", roi)
        # cv2.imshow("ThresholdWhite", frame_threshed_WHITE)
        # cv2.imshow("ThresholdBlue", frame_threshed_BLUE)
        # cv2.imshow("ThresholdBlack", frame_threshed_BLACK)
        cv2.imshow('VideoFeed',frame)
        
        # rawCapture.truncate(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
    