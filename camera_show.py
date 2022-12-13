import cv2
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
cv2.imshow("VideoFeed", frame)
while True:
    ret, frame = camera.read()
    cv2.imshow("VideoFeed", frame)  
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()