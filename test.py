import cv2,cv
import numpy as np

# create video capture
cap = cv2.VideoCapture("/Users/hayer/Desktop/Anand/openfields/071211_Batch1-openfield.m4v")
##if (!cap.isOpened()):  // check if we succeeded
 ##   return -1;
print cap.isOpened()

while(1):

    # read the frames
    _,frame = cap.read()
    capture = cv.CaptureFromFile("/Users/hayer/Desktop/Anand/openfields/071211_Batch1-openfield.m4v")

    # smooth it
    #cv2.imshow('before',frame)
    frame = cv2.blur(frame,(17,17))
    #cv2.imshow('blur',frame)
    # convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv,np.array((0, 0, 1)), np.array((0, 0, 4)))

    #res = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,element)
    #cv2.imshow('thresh_first2',res)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
    cv2.dilate(thresh,element,thresh,None,10)
    cv2.imshow('thresh_first',thresh)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
    cv2.erode(thresh,element,thresh,None,2)
    cv2.imshow('thresh_2',thresh)
    #cv2.dilate(thresh,element,thresh,None,10)
    thresh2 = thresh.copy()

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # finding contour with maximum area and store it as best_cnt
    color_image = cv.QueryFrame(capture)
    max_area = 0
    contours2 = contours
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print len(cnt)
        cnt2 = cv2.convexHull(cnt)
        #print len(cnt2)
        contours2[i] = cnt2
        i = i+1
        if area > max_area:
            max_area = area
            best_cnt = cnt

    cv2.drawContours(frame,contours2,-1,cv.CV_RGB(255,255,0),1)
    #print contours
    # finding centroids of best_cnt and draw a circle there
    #M = cv2.moments(best_cnt)
    #cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    #cv2.circle(frame,(cx,cy),5,255,-1)

    # Show it, if key pressed is 'Esc', exit the loop
    cv2.imshow('frame',frame)
    cv2.imshow('thresh',thresh2)
    cv2.imshow('contour',frame)
    if cv2.waitKey(33)== 27:
        break

# Clean up everything before leaving
cv2.destroyAllWindows()
cap.release()
