import cv2,cv
import numpy as np
import matplotlib.pyplot as plt

CV_CAP_PROP_POS_MSEC = 0
CV_CAP_PROP_POS_FRAMES = 1
CV_CAP_PROP_POS_AVI_RATIO = 2
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FOURCC = 6
CV_CAP_PROP_FRAME_COUNT = 7
CV_CAP_PROP_FORMAT = 8
CV_CAP_PROP_MODE = 9
#CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
#CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
#CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
#CV_CAP_PROP_HUE Hue of the image (only for cameras).
#CV_CAP_PROP_GAIN Gain of the image (only for cameras).
#CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
#CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
#CV_CAP_PROP_WHITE_BALANCE Currently not supported
#CV_CAP_PROP_RECTIFICATION

# create video capture
cap = cv2.VideoCapture("/Users/hayer/Desktop/Anand/openfields/071211_Batch1-openfield.m4v")
##if (!cap.isOpened()):  // check if we succeeded
 ##   return -1;
print cap.isOpened()

total_number_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT)
print total_number_of_frames
width = int(cap.get(3))
height = int(cap.get(4))


#import heatmap
#import random
#
#hm = heatmap.Heatmap()
#pts = [(random.uniform(-77.012, -77.050), random.uniform(38.888, 38.910)) for x in range(100)]
#hm.heatmap(pts)
#hm.saveKML("data.kml")

#print frame
#width = frame.shape[0]
#height = frame.shape[1]
#print width
#print height

half_width = int(width/2)
half_height = int(height/2)

perc_75_width = int((half_width-half_width*0.55)/2)
perc_75_height = int((half_height-half_height*0.55)/2)
perc_50_width = int((half_width-half_width*0.35)/2)
perc_50_height = int((half_height-half_height*0.35)/2)
perc_25_width = int((half_width-half_width*0.15)/2)
perc_25_height = int((half_height-half_height*0.15)/2)

upper_left_point1 = tuple([0,0])
upper_left_point2 = tuple([half_width,half_height])
upper_left_75_point1 = tuple([perc_75_width+15,perc_75_height+5])
upper_left_75_point2 = tuple([half_width-perc_75_width+15,half_height-perc_75_height-5])
upper_left_50_point1 = tuple([perc_50_width+15,perc_50_height+5])
upper_left_50_point2 = tuple([half_width-perc_50_width+15,half_height-perc_50_height-5])
upper_left_25_point1 = tuple([perc_25_width+15,perc_25_height+5])
upper_left_25_point2 = tuple([half_width-perc_25_width+15,half_height-perc_25_height-5])
upper_right_point1 = tuple([half_width,0])
upper_right_point2 = tuple([width,half_height])
upper_right_75_point1 = tuple([half_width+perc_75_width-15,perc_75_height+5])
upper_right_75_point2 = tuple([width-perc_75_width-15,half_height-perc_75_height-5])
upper_right_50_point1 = tuple([half_width+perc_50_width-15,perc_50_height+5])
upper_right_50_point2 = tuple([width-perc_50_width-15,half_height-perc_50_height-5])
upper_right_25_point1 = tuple([half_width+perc_25_width-15,perc_25_height+5])
upper_right_25_point2 = tuple([width-perc_25_width-15,half_height-perc_25_height-5])


_,frame2 = cap.read()

x = []
y = []

# convert to hsv and find range of colors
#hsv = cv2.cvtColor(frame_cross,cv2.COLOR_BGR2HSV)
#thresh = cv2.inRange(hsv,np.array((250, 250, 250)), np.array((255,255,255)))
#cv2.imshow('blur',frame_cross)
print half_width
print half_height
frame_number = cap.get(CV_CAP_PROP_POS_FRAMES)
while(frame_number < 5000):
    frame_number = cap.get(CV_CAP_PROP_POS_FRAMES)
    # read the frames
    _,frame = cap.read()
    #capture = cv.CaptureFromFile("/Users/hayer/Desktop/Anand/openfields/071211_Batch1-openfield.m4v")

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
    #cv2.imshow('thresh_first',thresh)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
    cv2.erode(thresh,element,thresh,None,2)
    #cv2.imshow('thresh_2',thresh)
    #cv2.dilate(thresh,element,thresh,None,10)
    thresh2 = thresh.copy()

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # finding contour with maximum area and store it as best_cnt
    _,color_image = cap.read()
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
    start = tuple([0,half_height])
    end = tuple([width,half_height])
    cv2.line(frame,start,end,cv.CV_RGB(255,0,255))
    start = tuple([half_width,0])
    end = tuple([half_width,height])
    cv2.line(frame,start,end,cv.CV_RGB(255,0,255))
    cv2.line(frame,start,end,cv.CV_RGB(255,0,255))
    cv2.rectangle(frame,upper_left_75_point1,upper_left_75_point2,cv.CV_RGB(255,0,255))
    cv2.rectangle(frame,upper_left_50_point1,upper_left_50_point2,cv.CV_RGB(255,0,255))
    cv2.rectangle(frame,upper_left_25_point1,upper_left_25_point2,cv.CV_RGB(255,0,255))
    cv2.rectangle(frame,upper_right_75_point1,upper_right_75_point2,cv.CV_RGB(255,0,0))
    cv2.rectangle(frame,upper_right_50_point1,upper_right_50_point2,cv.CV_RGB(255,0,0))
    cv2.rectangle(frame,upper_right_25_point1,upper_right_25_point2,cv.CV_RGB(255,0,0))

    for cnt in contours2:
        summe = [0,0]
        number_of_points = 0
        for point in cnt:
            #print point
            # print point[0][1]
            number_of_points = number_of_points + 1
            summe = [summe[0]+point[0][0],summe[1]+point[0][1]]
        mid_point = [summe[0]/number_of_points,summe[1]/number_of_points]
        x.append(mid_point[0])
        y.append(mid_point[1])
        mid_point = tuple([int(mid_point[0]),int(mid_point[1])])
        cv2.circle(frame2,mid_point, 1, cv.CV_RGB(255,0,0))
        cv2.circle(frame,mid_point, 1, cv.CV_RGB(255,0,0))

    #print contours
    # finding centroids of best_cnt and draw a circle there
    #M = cv2.moments(best_cnt)
    #cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    #cv2.circle(frame,(cx,cy),5,255,-1)

    # Show it, if key pressed is 'Esc', exit the loop

    cv2.imshow('thresh',thresh2)
    cv2.imshow('contour',frame2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(33)== 27:
        break
cv2.imwrite("tra.png",frame2)
# Clean up everything before leaving
cv2.destroyAllWindows()
cap.release()


#
## Generate some test data
#
#
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
plt.clf()

plt.imshow(heatmap, extent=extent)
cb = plt.colorbar()
cb.set_label('mean value')
plt.savefig("heatmap.png")


