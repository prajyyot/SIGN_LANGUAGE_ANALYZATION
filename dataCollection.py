import time

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#Imported necessary modules for image capturing


cap = cv2.VideoCapture(0) #0 is id num for webcam
detector = HandDetector(maxHands=1) #will only need 1 hand so

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    #for cropping img
    if hands:
        hand = hands[0] #coz only one hand no other
        x,y,w,h = hand['bbox'] #bounding box

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

        imgCrop = img[y-offset: y + h+offset, x-offset: x + w+offset] #basically our pic is matrix so should give height and width

        aspectRatio = h/w
        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape

            #to centre the pic
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else: #doing for width
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape

            # to centre the pic
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize


        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite", imgWhite)



    cv2.imshow("Image",img)
    key = cv2.waitKey(1) #delay of 1 msec
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
