import cv2
import time
import Pose_Estimation_Module as pm

cap = cv2.VideoCapture('PoseVideos/Pose4.mp4')
ptime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.findPosition(img, draw= False)
    if len(lmlist) != 0:
        # print(lmlist)
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 20, (0, 0, 255), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    img = cv2.resize(img, (1080, 1920))
    print(img.shape)
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 700, 700)
    cv2.imshow("Resized_Window", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
