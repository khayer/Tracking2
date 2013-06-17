import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import math

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
cap = cv2.VideoCapture("/Users/hayer/Desktop/Anand/openfields/100411_batch2-openfield.m4v")
##if (!cap.isOpened()):  // check if we succeeded
 ##   return -1;
print cap.isOpened()

total_number_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT)
print total_number_of_frames
frame_per_sec = cap.get(CV_CAP_PROP_FPS)
print frame_per_sec
width = int(cap.get(3))
height = int(cap.get(4))

def comp_tuple(mp,pt1,pt2):
    return mp[0] >= pt1[0] and mp[0] <= pt2[0] and mp[1] >= pt1[1] and mp[1] <= pt2[1]

def dist(pt1,pt2):
    xd = pt2[0] - pt1[0]
    yd = pt2[1] - pt1[1]
    return math.sqrt(xd*xd + yd*yd)

half_width = int(width/2)
half_height = int(height/2)

perc_75_width = int((half_width-half_width*0.55)/2)
perc_75_height = int((half_height-half_height*0.55)/2)
perc_50_width = int((half_width-half_width*0.35)/2)
perc_50_height = int((half_height-half_height*0.35)/2)
perc_25_width = int((half_width-half_width*0.15)/2)
perc_25_height = int((half_height-half_height*0.15)/2)

threshold = 5
correction = 35
correction_height = 5 # old
correction_height_inner = 10
correction_height_outer = 20


last_upper_left = None
upper_left = upper_left_75 = upper_left_50 = upper_left_25 = 0
dist_upper_left = dist_upper_left_75 = dist_upper_left_50 = dist_upper_left_25 = 0
upper_left_point1 = tuple([0,0])
upper_left_point2 = tuple([half_width,half_height])
upper_left_75_point1 = tuple([perc_75_width+correction,perc_75_height+correction_height])
upper_left_75_point2 = tuple([half_width-perc_75_width+correction,half_height-perc_75_height+correction_height_outer])
upper_left_50_point1 = tuple([perc_50_width+correction,perc_50_height+correction_height])
upper_left_50_point2 = tuple([half_width-perc_50_width+correction,half_height-perc_50_height+correction_height_outer])
upper_left_25_point1 = tuple([perc_25_width+correction,perc_25_height+correction_height])
upper_left_25_point2 = tuple([half_width-perc_25_width+correction,half_height-perc_25_height+correction_height_outer])

last_upper_right = None
upper_right = upper_right_75 = upper_right_50 = upper_right_25 = 0
dist_upper_right = dist_upper_right_75 = dist_upper_right_50 = dist_upper_right_25 = 0
upper_right_point1 = tuple([half_width,0])
upper_right_point2 = tuple([width,half_height])
upper_right_75_point1 = tuple([half_width+perc_75_width-correction,perc_75_height+correction_height])
upper_right_75_point2 = tuple([width-perc_75_width-correction,half_height-perc_75_height+correction_height_outer])
upper_right_50_point1 = tuple([half_width+perc_50_width-correction,perc_50_height+correction_height])
upper_right_50_point2 = tuple([width-perc_50_width-correction,half_height-perc_50_height+correction_height_outer])
upper_right_25_point1 = tuple([half_width+perc_25_width-correction,perc_25_height+correction_height])
upper_right_25_point2 = tuple([width-perc_25_width-correction,half_height-perc_25_height+correction_height_outer])

lower_left = lower_left_75 = lower_left_50 = lower_left_25 = 0
lower_left_point1 = tuple([0,half_height])
lower_left_point2 = tuple([half_width,height])
lower_left_75_point1 = tuple([perc_75_width+correction,half_height+perc_75_height+correction_height])
lower_left_75_point2 = tuple([half_width-perc_75_width+correction,height-perc_75_height-correction_height_outer])
lower_left_50_point1 = tuple([perc_50_width+correction,half_height+perc_50_height+correction_height])
lower_left_50_point2 = tuple([half_width-perc_50_width+correction,height-perc_50_height-correction_height_outer])
lower_left_25_point1 = tuple([perc_25_width+correction,half_height+perc_25_height+correction_height])
lower_left_25_point2 = tuple([half_width-perc_25_width+correction,height-perc_25_height-correction_height_outer])

lower_right = lower_right_75 = lower_right_50 = lower_right_25 = 0
lower_right_point1 = tuple([half_width,half_height])
lower_right_point2 = tuple([width,height])
lower_right_75_point1 = tuple([half_width+perc_75_width-correction,half_height+perc_75_height-correction_height_inner])
lower_right_75_point2 = tuple([width-perc_75_width-correction,height-perc_75_height-correction_height_outer])
lower_right_50_point1 = tuple([half_width+perc_50_width-correction,half_height+perc_50_height-correction_height_inner])
lower_right_50_point2 = tuple([width-perc_50_width-correction,height-perc_50_height-correction_height_outer])
lower_right_25_point1 = tuple([half_width+perc_25_width-correction,half_height+perc_25_height-correction_height_inner])
lower_right_25_point2 = tuple([width-perc_25_width-correction,height-perc_25_height-correction_height_outer])
_,frame2 = cap.read()

x = []
y = []
mid_points = []

# convert to hsv and find range of colors
#hsv = cv2.cvtColor(frame_cross,cv2.COLOR_BGR2HSV)
#thresh = cv2.inRange(hsv,np.array((250, 250, 250)), np.array((255,255,255)))
#cv2.imshow('blur',frame_cross)
print half_width
print half_height
frame_number = cap.get(CV_CAP_PROP_POS_FRAMES)
while(frame_number < 100):
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
    cv2.rectangle(frame,lower_left_75_point1,lower_left_75_point2,cv.CV_RGB(0,0,255))
    cv2.rectangle(frame,lower_left_50_point1,lower_left_50_point2,cv.CV_RGB(0,0,255))
    cv2.rectangle(frame,lower_left_25_point1,lower_left_25_point2,cv.CV_RGB(0,0,255))
    cv2.rectangle(frame,lower_right_75_point1,lower_right_75_point2,cv.CV_RGB(0,255,0))
    cv2.rectangle(frame,lower_right_50_point1,lower_right_50_point2,cv.CV_RGB(0,255,0))
    cv2.rectangle(frame,lower_right_25_point1,lower_right_25_point2,cv.CV_RGB(0,255,0))

    mid_points = []
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
        mid_points.append(mid_point)
        cv2.circle(frame2,mid_point, 1, cv.CV_RGB(255,0,0))
        cv2.circle(frame,mid_point, 1, cv.CV_RGB(255,0,0))

    for mp in mid_points:

        if comp_tuple(mp,upper_left_point1,upper_left_point2):
            #var1 = 4 if var1 is None else var1
            last_upper_left = last_upper_left or mp
            distance = dist(last_upper_left,mp)
            if distance > threshold:
                cv2.circle(frame,mp, 5, cv.CV_RGB(255,0,0))
            else:
                distance = 0

            if comp_tuple(mp,upper_left_25_point1,upper_left_25_point2):
                dist_upper_left_25 = dist_upper_left_25 + distance
                upper_left_25 = upper_left_25 + 1
            elif comp_tuple(mp,upper_left_50_point1,upper_left_50_point2):
                upper_left_50 = upper_left_50 + 1
                dist_upper_left_50 = dist_upper_left_50 + distance
            elif comp_tuple(mp,upper_left_75_point1,upper_left_75_point2):
                upper_left_75 = upper_left_75 + 1
                dist_upper_left_75 = dist_upper_left_75 + distance
            else:
                upper_left = upper_left + 1
                dist_upper_left = dist_upper_left + distance
            last_upper_left = mp

        if comp_tuple(mp,upper_right_point1,upper_right_point2):
            last_upper_right = last_upper_right or mp
            distance = dist(last_upper_right,mp)
            if distance > threshold:
                cv2.circle(frame,mp, 5, cv.CV_RGB(0,0,255))
            else:
                distance = 0
            if comp_tuple(mp,upper_right_25_point1,upper_right_25_point2):
                dist_upper_right_25 = dist_upper_right_25 + distance
                upper_right_25 = upper_right_25 + 1
            elif comp_tuple(mp,upper_right_50_point1,upper_right_50_point2):
                upper_right_50 = upper_right_50 + 1
                dist_upper_right_50 = dist_upper_right_50 + distance
            elif comp_tuple(mp,upper_right_75_point1,upper_right_75_point2):
                upper_right_75 = upper_right_75 + 1
                dist_upper_right_75 = dist_upper_right_75 + distance
            else:
                upper_right = upper_right + 1
                dist_upper_right = dist_upper_right + distance
            last_upper_right = mp

        if comp_tuple(mp,lower_left_point1,lower_left_point2):
            cv2.circle(frame,mp, 5, cv.CV_RGB(0,255,0))
            if comp_tuple(mp,lower_left_25_point1,lower_left_25_point2):
                lower_left_25 = lower_left_25 + 1
            elif comp_tuple(mp,lower_left_50_point1,lower_left_50_point2):
                lower_left_50 = lower_left_50 + 1
            elif comp_tuple(mp,lower_left_75_point1,lower_left_75_point2):
                lower_left_75 = lower_left_75 + 1
            else:
                lower_left = lower_left + 1

        if comp_tuple(mp,lower_right_point1,lower_right_point2):
            cv2.circle(frame,mp, 5, cv.CV_RGB(0,255,255))
            if comp_tuple(mp,lower_right_25_point1,lower_right_25_point2):
                lower_right_25 = lower_right_25 + 1
            elif comp_tuple(mp,lower_right_50_point1,lower_right_50_point2):
                lower_right_50 = lower_right_50 + 1
            elif comp_tuple(mp,lower_right_75_point1,lower_right_75_point2):
                lower_right_75 = lower_right_75 + 1
            else:
                lower_right = lower_right + 1


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
print "upper_left:\t" + str(upper_left)
print "upper_left_25:\t" + str(upper_left_25)
print "upper_left_50:\t" + str(upper_left_50)
print "upper_left_75:\t" + str(upper_left_75)

print "dist_upper_left:\t" + str(dist_upper_left)
print "dist_upper_left_25:\t" + str(dist_upper_left_25)
print "dist_upper_left_50:\t" + str(dist_upper_left_50)
print "dist_upper_left_75:\t" + str(dist_upper_left_75)

print "upper_right:\t" + str(upper_right)
print "upper_right_25:\t" + str(upper_right_25)
print "upper_right_50:\t" + str(upper_right_50)
print "upper_right_75:\t" + str(upper_right_75)

print "dist_upper_right:\t" + str(dist_upper_right)
print "dist_upper_right_25:\t" + str(dist_upper_right_25)
print "dist_upper_right_50:\t" + str(dist_upper_right_50)
print "dist_upper_right_75:\t" + str(dist_upper_right_75)

print "lower_left:\t" + str(lower_left)
print "lower_left_25:\t" + str(lower_left_25)
print "lower_left_50:\t" + str(lower_left_50)
print "lower_left_75:\t" + str(lower_left_75)

print "lower_right:\t" + str(lower_right)
print "lower_right_25:\t" + str(lower_right_25)
print "lower_right_50:\t" + str(lower_right_50)
print "lower_right_75:\t" + str(lower_right_75)

plt.imshow(heatmap, extent=extent)
cb = plt.colorbar()
cb.set_label('mean value')
plt.savefig("heatmap.png")


