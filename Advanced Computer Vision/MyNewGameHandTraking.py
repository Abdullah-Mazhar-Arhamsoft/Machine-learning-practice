{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import mediapipe as mp\n",
    "import time\n",
    "import Hand_Tracking_Module as htm\n",
    "\n",
    "\n",
    "pTime = 0\n",
    "cTime = 0\n",
    "cap = cv2.VideoCapture(0)\n",
    "detector = handDetector()\n",
    "\n",
    "while True:\n",
    "    success, img = cap.read()\n",
    "    img = detector.findHands(img)\n",
    "    lmlist = detector.findPosition(img)\n",
    "    if len(lmlist) != 0:\n",
    "        print(lmlist[4])\n",
    "\n",
    "    cTime = time.time()\n",
    "    fps = 1 / (cTime - pTime)\n",
    "    pTime = cTime\n",
    "\n",
    "    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)\n",
    "\n",
    "    cv2.imshow(\"Image\", img)\n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "        break"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "env1",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.7 (tags/v3.10.7:6cc6b13, Sep  5 2022, 14:08:36) [MSC v.1933 64 bit (AMD64)]"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "9762b65ea092d07537403797cd77c77172dbb418b70d6a03e78e0a3e8fcb34d1"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
