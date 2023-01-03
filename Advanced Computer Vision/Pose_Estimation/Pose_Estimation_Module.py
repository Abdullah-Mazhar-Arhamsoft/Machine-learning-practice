import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode = False, smooth = True, DetectionCon = 0.5, TrackCon = 0.5):
        self.mode = mode
        
        self.smooth = smooth
        self.DetectionCon = DetectionCon
        self.TrackCon = TrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = self.mode,
                                     smooth_landmarks = self.smooth, min_detection_confidence = self.DetectionCon,
                                      min_tracking_confidence = self.TrackCon)


    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img 
        
    def findPosition(self, img, draw = True):
        lmlist = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w) , int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmlist


# def main():
detector = poseDetector()
cap = cv2.VideoCapture('Pose_Estimation/PoseVideos/Pose3.mp4')
ptime = 0

while (cap.isOpened()):
    success, img = cap.read()
    # print(img)
    img = detector.findPose(img)
    lmlist = detector.findPosition(img, draw= False)
    if len(lmlist) != 0 :
        # print(lmlist)
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 12, (0, 0, 255), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    img = cv2.resize(img, (1080, 1920))
    # print(img)
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 700, 700)
    cv2.imshow("Resized_Window", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# if __name__ == "__main__":
#     main()