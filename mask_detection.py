
import cv2
import time
import numpy as np

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
    

def main():
    min_contour_area = 500
    # avgFrame = None
    # resolution = (640, 480)
    # fps = 32

    # camera = PiCamera()
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    cv2.imshow("VideoFeed", frame)
    # camera.resolution = resolution
    # camera.framerate = fps
    # rawCapture = PiRGBArray(camera, size=resolution)
    time.sleep(1)

    print("Starting video capture, press 'q' to exit")

    # H = 0-180, S = 0-255, V = 0-255
    # /360*180, /100*255, /100*255
    COLOR_BLUE_MIN = np.array([118,0,75])
    COLOR_BLUE_MAX = np.array([254,30,100])

    COLOR_BLACK_MIN = np.array([0,0,0])
    COLOR_BLACK_MAX = np.array([359,50,30])

    COLOR_WHITE_MIN = np.array([0,0,75])
    COLOR_WHITE_MAX = np.array([359,10,100])

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

    # for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    i = 0
    while True:
        ret, frame = camera.read()

        # frame = f.array
        # resize frame since processing on large raw image is costly/unnecessary
        # frame = imutils.resize(frame, width=500)
        cv2.resize(frame, (500,500), interpolation = cv2.INTER_AREA)
        blurredFrame = cv2.GaussianBlur(frame, (25, 25), 0)
        roi = blurredFrame[160:-190, 200:-200, :]
        roiHSV = np.zeros(shape = roi.shape)
        roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # H_min = cv2.getTrackbarPos('H_min', 'Controls')
        # S_min = cv2.getTrackbarPos('S_min', 'Controls')
        # V_min = cv2.getTrackbarPos('V_min', 'Controls')
        # H_max = cv2.getTrackbarPos('H_max', 'Controls')
        # S_max = cv2.getTrackbarPos('S_max', 'Controls')
        # V_max = cv2.getTrackbarPos('V_max', 'Controls')


        frame_threshed_BLUE = cv2.inRange(roiHSV, (COLOR_BLUE_MIN[0]/360*180,COLOR_BLUE_MIN[1]/100*255,COLOR_BLUE_MIN[2]/100*255), (COLOR_BLUE_MAX[0]/360*180,COLOR_BLUE_MAX[1]/100*255,COLOR_BLUE_MAX[2]/100*255))
        frame_threshed_WHITE = cv2.inRange(roiHSV, (COLOR_WHITE_MIN[0]/360*180,COLOR_WHITE_MIN[1]/100*255,COLOR_WHITE_MIN[2]/100*255), (COLOR_WHITE_MAX[0]/360*180,COLOR_WHITE_MAX[1]/100*255,COLOR_WHITE_MAX[2]/100*255))
        frame_threshed_BLACK = cv2.inRange(roiHSV, (COLOR_BLACK_MIN[0]/360*180,COLOR_BLACK_MIN[1]/100*255,COLOR_BLACK_MIN[2]/100*255), (COLOR_BLACK_MAX[0]/360*180,COLOR_BLACK_MAX[1]/100*255,COLOR_BLACK_MAX[2]/100*255))
        
        frame_threshed_BLUE = cv2.dilate(frame_threshed_BLUE, None, iterations=1)
        frame_threshed_WHITE = cv2.dilate(frame_threshed_WHITE, None, iterations=1)
        frame_threshed_BLACK = cv2.dilate(frame_threshed_BLACK, None, iterations=1)

        # frame_threshed = cv2.bitwise_or(frame_threshed_BLUE, frame_threshed_WHITE)
        # frame_threshed = cv2.bitwise_or(frame_threshed, frame_threshed_BLACK)
        contours_BLUE = cv2.findContours(frame_threshed_BLUE.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours_WHITE = cv2.findContours(frame_threshed_WHITE.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] 
        contours_BLACK = cv2.findContours(frame_threshed_BLACK.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]  
        # contours = cv2.findContours(frame_threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] 


        for (contour_BLUE, contour_WHITE, contour_BLACK) in zip(contours_BLUE, contours_WHITE, contours_BLACK):
            if cv2.contourArea(contour_BLUE) < min_contour_area:
                continue

            (x1, y1, w1, h1) = cv2.boundingRect(contour_BLUE)
            (x2, y2, w2, h2) = cv2.boundingRect(contour_WHITE)
            (x3, y3, w3, h3) = cv2.boundingRect(contour_BLACK)

            # print(x)
            # print(y)
            cv2.rectangle(frame, (200+x1 + 10,160+y1 + 10), (x1 + w1 + 200 + 10, y1 + 160 + h1 + 10), (0, 0, 255), 2)
            cv2.rectangle(frame, (200+x2 + 10,160+y2 + 10), (x2 + w2 + 200 + 10, y2 + 160 + h2 + 10), (0, 0, 255), 2)
            cv2.rectangle(frame, (200+x3 + 10,160+y3 + 10), (x3 + w3 + 200 + 10, y3 + 160 + h3 + 10), (0, 0, 255), 2)
            # cv2.rectangle(frame_threshed, (x+10, y+10), (x + w, y + h), (255, 255, 255), 2)

        # cv2.putText(frame, motionStatus, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        maskOn = False
        if detectMask(frame_threshed_WHITE) or detectMask(frame_threshed_BLUE) or detectMask(frame_threshed_BLACK):
            maskOn = True

        cv2.putText(frame, str(maskOn), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("VideoFeed", frame)
        cv2.imshow("Blurred", blurredFrame[160:-190, 200:-200, :])
        cv2.imshow("ThresholdWhite", frame_threshed_WHITE)
        cv2.imshow("ThresholdBlue", frame_threshed_BLUE)
        cv2.imshow("ThresholdBlack", frame_threshed_BLACK)

        

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
    