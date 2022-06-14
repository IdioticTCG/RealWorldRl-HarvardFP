import cv2

cam = cv2.VideoCapture(1)
 
cv2.namedWindow("test")
 
img_counter = 0
frame = None
 
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
 
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
 
cam.release()
 
cv2.destroyAllWindows()
print(min(frame.flatten()))