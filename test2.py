import cv2,cv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from xlutils.copy import copy
from xlrd import open_workbook
from xlwt import easyxf

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

if len(sys.argv) != 3:
    sys.stderr.write("usage: python test.py video_file results.xls\n")
    sys.exit()
# create video capture
#cap = cv2.VideoCapture("/Users/hayer/Desktop/Anand/openfields/100611_openfield-b5.m4v")
#cap = cv2.VideoCapture("/Users/kat/Desktop/071411_batch4-openfield.m4v")
cap = cv2.VideoCapture(sys.argv[1])
sample_name = sys.argv[1].split("/")[-1].split(".")[0]


print sample_name


##if (!cap.isOpened()):  // check if we succeeded
 ##   return -1;
print cap.isOpened()

total_number_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT)
print "Total number of frames"
print total_number_of_frames
frame_per_sec = 30.03888889
print "Frame per sec"
print frame_per_sec

_,frame2 = cap.read()
#frame2 = cv2.blur(frame2,(17,17))
imgray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,200,255,0)
cv2.imwrite("gray_tra.png",thresh)
cv2.imshow('thresh',thresh)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
num_p = 0
all_x = []
all_y = []
while not contours:
    _,frame2 = cap.read()
    cv2.imshow('frame2',frame2)
    #frame2 = cv2.blur(frame2,(17,17))
    imgray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgray',imgray)
    ret,thresh = cv2.threshold(imgray,200,255,0)
    #cv2.imwrite("gray_tra.png",thresh)
    #cv2.imshow('thresh',thresh)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    for point in cnt:
        num_p = num_p + 1
        x = point[0][0]
        y = point[0][1]
        if x > 20 and x < 640 and y > 20 and y < 460:
            all_x.append(x)
            all_y.append(y)
print (all_x)
print (all_y)
print np.median(all_x)
print np.median(all_y)


actual_width = int(cap.get(3))
actual_height = int(cap.get(4))

width = 2*int(np.median(all_x))
height = 2*int(np.median(all_y))

diff_width = actual_width - width
diff_height = actual_height - height


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

bout_threshold = 11
dist_threshold = 3
conversion_pixel_to_cm = 10
print diff_width
print diff_height
if diff_width <= 0:
    correction_width_inner = int(-diff_width*20)
    correction_width_outer = int(-diff_width*20)
elif diff_width < 10:
    correction_width_inner = int(diff_width*10)
    correction_width_outer = int(diff_width*10)
elif diff_width >= 10:
    correction_width_inner = int(diff_width)
    correction_width_outer = int(diff_width)
else:
    correction_width_inner = int(diff_width*1.5)
    correction_width_outer = int(diff_width*1.7)
if diff_height < 10:
    correction_height_inner = -diff_height
    correction_height_outer = -diff_height
elif diff_height < 0:
    correction_height_inner = -diff_height*2
    correction_height_outer = -diff_height*2
elif diff_height > 10 and diff_height <= 30:
    correction_height_inner = -diff_height/2
    correction_height_outer = -diff_height/2
elif diff_height > 30:
    correction_height_inner = diff_height/10
    correction_height_outer = -diff_height/5
else:
    correction_height_inner = diff_height
    correction_height_outer = diff_height

last_upper_left = None
upper_left_i = None
upper_left_num_bout = 0
upper_left_frame_bout = 0
upper_left_distance_bout = 0
upper_left_lap_bout = []
upper_left = upper_left_75 = upper_left_50 = upper_left_25 = 0
dist_upper_left = dist_upper_left_75 = dist_upper_left_50 = dist_upper_left_25 = 0
upper_left_point1 = tuple([0,0])
upper_left_point2 = tuple([half_width,half_height])
upper_left_last_known_point = upper_left_point1
upper_left_75_point1 = tuple([perc_75_width+ correction_width_outer,perc_75_height+correction_height_outer])
upper_left_75_point2 = tuple([half_width-perc_75_width+ correction_width_inner,half_height-perc_75_height+correction_height_inner])
upper_left_50_point1 = tuple([perc_50_width+ correction_width_outer,perc_50_height+correction_height_outer])
upper_left_50_point2 = tuple([half_width-perc_50_width+ correction_width_inner,half_height-perc_50_height+correction_height_inner])
upper_left_25_point1 = tuple([perc_25_width+ correction_width_outer,perc_25_height+correction_height_outer])
upper_left_25_point2 = tuple([half_width-perc_25_width+ correction_width_inner,half_height-perc_25_height+correction_height_inner])

last_upper_right = None
upper_right_i = None
upper_right_num_bout = 0
upper_right_frame_bout = 0
upper_right_distance_bout = 0
upper_right_lap_bout = []
upper_right = upper_right_75 = upper_right_50 = upper_right_25 = 0
dist_upper_right = dist_upper_right_75 = dist_upper_right_50 = dist_upper_right_25 = 0
upper_right_point1 = tuple([half_width,0])
upper_right_point2 = tuple([width,half_height])
upper_right_last_known_point = upper_right_point1
upper_right_75_point1 = tuple([half_width+perc_75_width- correction_width_inner,perc_75_height+correction_height_outer])
upper_right_75_point2 = tuple([width-perc_75_width- correction_width_outer,half_height-perc_75_height+correction_height_inner])
upper_right_50_point1 = tuple([half_width+perc_50_width- correction_width_inner,perc_50_height+correction_height_outer])
upper_right_50_point2 = tuple([width-perc_50_width- correction_width_outer,half_height-perc_50_height+correction_height_inner])
upper_right_25_point1 = tuple([half_width+perc_25_width- correction_width_inner,perc_25_height+correction_height_outer])
upper_right_25_point2 = tuple([width-perc_25_width- correction_width_outer,half_height-perc_25_height+correction_height_inner])

last_lower_left = None
lower_left_i = None
lower_left_num_bout = 0
lower_left_frame_bout = 0
lower_left_distance_bout = 0
lower_left_lap_bout = []
lower_left = lower_left_75 = lower_left_50 = lower_left_25 = 0
dist_lower_left = dist_lower_left_75 = dist_lower_left_50 = dist_lower_left_25 = 0
lower_left_point1 = tuple([0,half_height])
lower_left_point2 = tuple([half_width,height])
lower_left_last_known_point = lower_left_point1
lower_left_75_point1 = tuple([perc_75_width+ correction_width_outer,half_height+perc_75_height-correction_height_inner])
lower_left_75_point2 = tuple([half_width-perc_75_width+ correction_width_inner,height-perc_75_height-correction_height_outer])
lower_left_50_point1 = tuple([perc_50_width+ correction_width_outer,half_height+perc_50_height-correction_height_inner])
lower_left_50_point2 = tuple([half_width-perc_50_width+ correction_width_inner,height-perc_50_height-correction_height_outer])
lower_left_25_point1 = tuple([perc_25_width+ correction_width_outer,half_height+perc_25_height-correction_height_inner])
lower_left_25_point2 = tuple([half_width-perc_25_width+ correction_width_inner,height-perc_25_height-correction_height_outer])


last_lower_right = None
lower_right_i = None
lower_right_num_bout = 0
lower_right_frame_bout = 0
lower_right_distance_bout = 0
lower_right_lap_bout = []
lower_right = lower_right_75 = lower_right_50 = lower_right_25 = 0
dist_lower_right = dist_lower_right_75 = dist_lower_right_50 = dist_lower_right_25 = 0
lower_right_point1 = tuple([half_width,half_height])
lower_right_point2 = tuple([width,height])
lower_right_last_known_point = lower_right_point1
lower_right_75_point1 = tuple([half_width+perc_75_width- correction_width_inner,half_height+perc_75_height-correction_height_inner])
lower_right_75_point2 = tuple([width-perc_75_width- correction_width_outer ,height-perc_75_height-correction_height_outer])
lower_right_50_point1 = tuple([half_width+perc_50_width-correction_width_inner,half_height+perc_50_height-correction_height_inner])
lower_right_50_point2 = tuple([width-perc_50_width-correction_width_outer,height-perc_50_height-correction_height_outer])
lower_right_25_point1 = tuple([half_width+perc_25_width-correction_width_inner,half_height+perc_25_height-correction_height_inner])
lower_right_25_point2 = tuple([width-perc_25_width-correction_width_outer,height-perc_25_height-correction_height_outer])




## Points for heatmap
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
time_in_msec = 0
#frame_per_sec = 0
while(frame_number < total_number_of_frames):
#while(frame_number < 250):
    frame_number = cap.get(CV_CAP_PROP_POS_FRAMES)
    #CV_CAP_PROP_POS_MSEC
    #l = cap.get(CV_CAP_PROP_POS_MSEC)
    #if l <= 1000:
    #    print l
    #    frame_per_sec += 1.0
    # read the frames
    _,frame = cap.read()
    if cap.get(CV_CAP_PROP_POS_MSEC) > 0.0:
        time_in_msec = cap.get(CV_CAP_PROP_POS_MSEC)
    percent = frame_number/total_number_of_frames * 100
    l = int(percent/2)
    if l%2==0:
      sys.stderr.write("\r[%-50s] %d%%" % ('='*int(l), percent))
      sys.stderr.flush()
    #capture = cv.CaptureFromFile("/Users/hayer/Desktop/Anand/openfields/071211_Batch1-openfield.m4v")

    # smooth it
    #cv2.imshow('before',frame)
    frame3 = cv2.blur(frame,(17,17))
    #cv2.imshow('blur',frame)
    # convert to hsv and find range of colors
    hsv = cv2.cvtColor(frame3,cv2.COLOR_BGR2HSV)
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

    max_area = 0
    contours2 = contours
    #i = 0
    #for cnt in contours:
    #    area = cv2.contourArea(cnt)
    #    #print len(cnt)
    #    cnt2 = cv2.convexHull(cnt)
    #    #print len(cnt2)
    #    contours2[i] = cnt2
    #    i = i+1
    #    if area > max_area:
    #        max_area = area
    #        best_cnt = cnt

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
        ## Points for Histogramm
        x.append(mid_point[0])
        y.append(mid_point[1])
        mid_point = tuple([int(mid_point[0]),int(mid_point[1])])
        mid_points.append(mid_point)
        cv2.circle(frame2,mid_point, 1, cv.CV_RGB(255,0,0))
        cv2.circle(frame,mid_point, 1, cv.CV_RGB(255,0,0))

    left_upper_bool = left_lower_bool = right_upper_bool = right_lower_bool = 1
    for mp in mid_points:

        if comp_tuple(mp,upper_left_point1,upper_left_point2) and left_upper_bool == 1:
            #var1 = 4 if var1 is None else var1
            left_upper_bool = 0
            last_upper_left = last_upper_left or mp
            distance = dist(last_upper_left,mp)
            if distance > dist_threshold:
                upper_left_i = 0 or upper_left_i
                upper_left_i = upper_left_i + 1
                if upper_left_i == bout_threshold:
                    upper_left_num_bout = upper_left_num_bout + 1
                    upper_left_lap_bout.append(0)
                if upper_left_i >= bout_threshold:
                    cv2.circle(frame,mp, 5, cv.CV_RGB(255,0,0))
                    upper_left_frame_bout = upper_left_frame_bout + 1
                    upper_left_distance_bout = upper_left_distance_bout + distance
                    upper_left_lap_bout[-1] = upper_left_lap_bout[-1] +1
            else:
                distance = 0
                upper_left_i = 0

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
            upper_left_last_known_point = mp

        if comp_tuple(mp,upper_right_point1,upper_right_point2) and right_upper_bool ==1 :
            right_upper_bool = 0
            last_upper_right = last_upper_right or mp
            distance = dist(last_upper_right,mp)
            if distance > dist_threshold:
                upper_right_i = 0 or upper_right_i
                upper_right_i = upper_right_i + 1
                if upper_right_i == bout_threshold:
                    upper_right_num_bout = upper_right_num_bout + 1
                    upper_right_lap_bout.append(0)
                if upper_right_i >= bout_threshold:
                    cv2.circle(frame,mp, 5, cv.CV_RGB(255,0,0))
                    upper_right_frame_bout = upper_right_frame_bout + 1
                    upper_right_distance_bout = upper_right_distance_bout + distance
                    upper_right_lap_bout[-1] = upper_right_lap_bout[-1] +1
            else:
                distance = 0
                upper_right_i = 0
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
            upper_right_last_known_point = mp

        if comp_tuple(mp,lower_left_point1,lower_left_point2) and left_lower_bool ==1:
            left_lower_bool = 0
            last_lower_left = last_lower_left or mp
            distance = dist(last_lower_left,mp)
            if distance > dist_threshold:
                lower_left_i = 0 or lower_left_i
                lower_left_i = lower_left_i + 1
                if lower_left_i == bout_threshold:
                    lower_left_num_bout = lower_left_num_bout + 1
                    lower_left_lap_bout.append(0)
                if lower_left_i >= bout_threshold:
                    cv2.circle(frame,mp, 5, cv.CV_RGB(0,255,0))
                    lower_left_frame_bout = lower_left_frame_bout + 1
                    lower_left_distance_bout = lower_left_distance_bout + distance
                    lower_left_lap_bout[-1] = lower_left_lap_bout[-1] +1
            else:
                distance = 0
                lower_left_i = 0
            if comp_tuple(mp,lower_left_25_point1,lower_left_25_point2):
                dist_lower_left_25 = dist_lower_left_25 + distance
                lower_left_25 = lower_left_25 + 1
            elif comp_tuple(mp,lower_left_50_point1,lower_left_50_point2):
                dist_lower_left_50 = dist_lower_left_50 + distance
                lower_left_50 = lower_left_50 + 1
            elif comp_tuple(mp,lower_left_75_point1,lower_left_75_point2):
                dist_lower_left_75 = dist_lower_left_75 + distance
                lower_left_75 = lower_left_75 + 1
            else:
                lower_left = lower_left + 1
                dist_lower_left = dist_lower_left + distance
            last_lower_left = mp
            lower_left_last_known_point = mp

        if comp_tuple(mp,lower_right_point1,lower_right_point2) and right_lower_bool:
            right_lower_bool = 0
            last_lower_right = last_lower_right or mp
            distance = dist(last_lower_right,mp)
            if distance > dist_threshold:
                lower_right_i = 0 or lower_right_i
                lower_right_i = lower_right_i + 1
                if lower_right_i == bout_threshold:
                    lower_right_num_bout = lower_right_num_bout + 1
                    lower_right_lap_bout.append(0)
                if lower_right_i >= bout_threshold:
                    cv2.circle(frame,mp, 5, cv.CV_RGB(255,255,0))
                    lower_right_frame_bout = lower_right_frame_bout + 1
                    lower_right_distance_bout = lower_right_distance_bout + distance
                    lower_right_lap_bout[-1] = lower_right_lap_bout[-1] +1
            else:
                distance = 0
                lower_right_i = 0
            if comp_tuple(mp,lower_right_25_point1,lower_right_25_point2):
                dist_lower_right_25 = dist_lower_right_25 + distance
                lower_right_25 = lower_right_25 + 1
            elif comp_tuple(mp,lower_right_50_point1,lower_right_50_point2):
                dist_lower_right_50 = dist_lower_right_50 + distance
                lower_right_50 = lower_right_50 + 1
            elif comp_tuple(mp,lower_right_75_point1,lower_right_75_point2):
                dist_lower_right_75 = dist_lower_right_75 + distance
                lower_right_75 = lower_right_75 + 1
            else:
                lower_right = lower_right + 1
                dist_lower_right = dist_lower_right + distance
            last_lower_right = mp
    if left_lower_bool == 1:
        mp = lower_left_last_known_point
        if comp_tuple(mp,lower_left_25_point1,lower_left_25_point2):
            lower_left_25 = lower_left_25 + 1
        elif comp_tuple(mp,lower_right_50_point1,lower_right_50_point2):
            lower_left_50 = lower_left_50 + 1
        elif comp_tuple(mp,lower_right_75_point1,lower_right_75_point2):
            lower_left_75 = lower_left_75 + 1
        else:
            lower_left = lower_left + 1
    if right_lower_bool == 1:
        mp = lower_right_last_known_point
        if comp_tuple(mp,lower_right_25_point1,lower_right_25_point2):
            lower_right_25 = lower_right_25 + 1
        elif comp_tuple(mp,lower_right_50_point1,lower_right_50_point2):
            lower_right_50 = lower_right_50 + 1
        elif comp_tuple(mp,lower_right_75_point1,lower_right_75_point2):
            lower_right_75 = lower_right_75 + 1
        else:
            lower_right = lower_right + 1
    if left_upper_bool == 1:
        mp = upper_left_last_known_point
        if comp_tuple(mp,upper_left_25_point1,upper_left_25_point2):
            upper_left_25 = upper_left_25 + 1
        elif comp_tuple(mp,upper_left_50_point1,upper_left_50_point2):
            upper_left_50 = upper_left_50 + 1
        elif comp_tuple(mp,upper_left_75_point1,upper_left_75_point2):
            upper_left_75 = upper_left_75 + 1
        else:
            upper_left = upper_left + 1
    if right_upper_bool == 1:
        mp = upper_right_last_known_point
        if comp_tuple(mp,upper_right_25_point1,upper_right_25_point2):
            upper_right_25 = upper_right_25 + 1
        elif comp_tuple(mp,upper_right_50_point1,upper_right_50_point2):
            upper_right_50 = upper_right_50 + 1
        elif comp_tuple(mp,upper_right_75_point1,upper_right_75_point2):
            upper_right_75 = upper_right_75 + 1
        else:
            upper_right = upper_right + 1
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

cv2.imwrite(sample_name + "_tra.png",frame2)
# Clean up everything before leaving
cv2.destroyAllWindows()


#all_frames = upper_left + upper_left_75 + upper_left_50 + upper_left_25
#frame_per_sec = 360000/all_frames
#print frame_per_sec
cap.release()


results_excel = sys.argv[2]
rb = open_workbook(results_excel,formatting_info=True)
r_sheet = rb.sheet_by_index(0)
wb = copy(rb) # a writable copy (I can't read values out of this, only write to it)
w_sheet = wb.get_sheet(0)
#for row_index in range(r_sheet.nrows,r_sheet.nrows+3):
#    #age_nov = r_sheet.cell(row_index, col_age_november).value
#    #If 3, then Combo I 3-4 year old  for both summer1 and fall1
#    w_sheet.write(row_index, 0, 'nina')
#    w_sheet.write(row_index, 1, 'k')


w_sheet.write(r_sheet.nrows+1,0,"Results for " + sample_name)
### UPPER LEFT
w_sheet.write(r_sheet.nrows+2,0,"upper_left")
# TIME
w_sheet.write(r_sheet.nrows+2,1,(upper_left    / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,2,(upper_left_75 / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,3,(upper_left_50 / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,4,(upper_left_25 / frame_per_sec))
# DISTANCE
w_sheet.write(r_sheet.nrows+2,5,(dist_upper_left /conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+2,6,(dist_upper_left_75 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+2,7,(dist_upper_left_50 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+2,8,(dist_upper_left_25 / conversion_pixel_to_cm))
# SPEED
speed_upper_left = speed_upper_left_25 = speed_upper_left_50 = speed_upper_left_75 = 0
if upper_left / frame_per_sec > 0:
    speed_upper_left = dist_upper_left/conversion_pixel_to_cm / ( upper_left /frame_per_sec)
if upper_left_25 / frame_per_sec > 0:
    speed_upper_left_25 = dist_upper_left_25/conversion_pixel_to_cm/ (upper_left_25/frame_per_sec)
if upper_left_50 / frame_per_sec > 0:
    speed_upper_left_50 = dist_upper_left_50/conversion_pixel_to_cm/ (upper_left_50/frame_per_sec )
if upper_left_75 / frame_per_sec > 0:
    speed_upper_left_75 = dist_upper_left_75/conversion_pixel_to_cm/ (upper_left_75/ frame_per_sec )
w_sheet.write(r_sheet.nrows+2,9,(speed_upper_left))
w_sheet.write(r_sheet.nrows+2,10,(speed_upper_left_75))
w_sheet.write(r_sheet.nrows+2,11,(speed_upper_left_50))
w_sheet.write(r_sheet.nrows+2,12,(speed_upper_left_25))
# BOUTS
w_sheet.write(r_sheet.nrows+2,13,(upper_left_num_bout))
w_sheet.write(r_sheet.nrows+2,14,(upper_left_frame_bout / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,15,(upper_left_distance_bout/conversion_pixel_to_cm))
for i,f in enumerate(upper_left_lap_bout):
    w_sheet.write(r_sheet.nrows+2,i+16,f)

### UPPER RIGHT
w_sheet.write(r_sheet.nrows+3,0,"upper_right")
# TIME
w_sheet.write(r_sheet.nrows+3,1,(upper_right    / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,2,(upper_right_75 / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,3,(upper_right_50 / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,4,(upper_right_25 / frame_per_sec))
# DISTANCE
w_sheet.write(r_sheet.nrows+3,5,(dist_upper_right /conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+3,6,(dist_upper_right_75 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+3,7,(dist_upper_right_50 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+3,8,(dist_upper_right_25 / conversion_pixel_to_cm))
# SPEED
speed_upper_right = speed_upper_right_25 = speed_upper_right_50 = speed_upper_right_75 = 0
if upper_right / frame_per_sec > 0:
    speed_upper_right = dist_upper_right/conversion_pixel_to_cm / ( upper_right /frame_per_sec)
if upper_right_25 / frame_per_sec > 0:
    speed_upper_right_25 = dist_upper_right_25/conversion_pixel_to_cm/ (upper_right_25/frame_per_sec)
if upper_right_50 / frame_per_sec > 0:
    speed_upper_right_50 = dist_upper_right_50/conversion_pixel_to_cm/ (upper_right_50/frame_per_sec )
if upper_right_75 / frame_per_sec > 0:
    speed_upper_right_75 = dist_upper_right_75/conversion_pixel_to_cm/ (upper_right_75/ frame_per_sec )
w_sheet.write(r_sheet.nrows+3,9,(speed_upper_right))
w_sheet.write(r_sheet.nrows+3,10,(speed_upper_right_75))
w_sheet.write(r_sheet.nrows+3,11,(speed_upper_right_50))
w_sheet.write(r_sheet.nrows+3,12,(speed_upper_right_25))
# BOUTS
w_sheet.write(r_sheet.nrows+3,13,(upper_right_num_bout))
w_sheet.write(r_sheet.nrows+3,14,(upper_right_frame_bout / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,15,(upper_right_distance_bout/conversion_pixel_to_cm))
for i,f in enumerate(upper_right_lap_bout):
    w_sheet.write(r_sheet.nrows+3,i+16,f)

### LOWER LEFT
w_sheet.write(r_sheet.nrows+4,0,"lower_left")
# TIME
w_sheet.write(r_sheet.nrows+4,1,(lower_left    / frame_per_sec))
w_sheet.write(r_sheet.nrows+4,2,(lower_left_75 / frame_per_sec))
w_sheet.write(r_sheet.nrows+4,3,(lower_left_50 / frame_per_sec))
w_sheet.write(r_sheet.nrows+4,4,(lower_left_25 / frame_per_sec))
# DISTANCE
w_sheet.write(r_sheet.nrows+4,5,(dist_lower_left /conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+4,6,(dist_lower_left_75 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+4,7,(dist_lower_left_50 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+4,8,(dist_lower_left_25 / conversion_pixel_to_cm))
# SPEED
speed_lower_left = speed_lower_left_25 = speed_lower_left_50 = speed_lower_left_75 = 0
if lower_left / frame_per_sec > 0:
    speed_lower_left = dist_lower_left/conversion_pixel_to_cm / ( lower_left /frame_per_sec)
if lower_left_25 / frame_per_sec > 0:
    speed_lower_left_25 = dist_lower_left_25/conversion_pixel_to_cm/ (lower_left_25/frame_per_sec)
if lower_left_50 / frame_per_sec > 0:
    speed_lower_left_50 = dist_lower_left_50/conversion_pixel_to_cm/ (lower_left_50/frame_per_sec )
if lower_left_75 / frame_per_sec > 0:
    speed_lower_left_75 = dist_lower_left_75/conversion_pixel_to_cm/ (lower_left_75/ frame_per_sec )
w_sheet.write(r_sheet.nrows+4,9,(speed_lower_left))
w_sheet.write(r_sheet.nrows+4,10,(speed_lower_left_75))
w_sheet.write(r_sheet.nrows+4,11,(speed_lower_left_50))
w_sheet.write(r_sheet.nrows+4,12,(speed_lower_left_25))
# BOUTS
w_sheet.write(r_sheet.nrows+4,13,(lower_left_num_bout))
w_sheet.write(r_sheet.nrows+4,14,(lower_left_frame_bout / frame_per_sec))
w_sheet.write(r_sheet.nrows+4,15,(lower_left_distance_bout/conversion_pixel_to_cm))
for i,f in enumerate(lower_left_lap_bout):
    w_sheet.write(r_sheet.nrows+4,i+16,f)

### UPPER LEFT
w_sheet.write(r_sheet.nrows+5,0,"lower_right")
# TIME
w_sheet.write(r_sheet.nrows+5,1,(lower_right    / frame_per_sec))
w_sheet.write(r_sheet.nrows+5,2,(lower_right_75 / frame_per_sec))
w_sheet.write(r_sheet.nrows+5,3,(lower_right_50 / frame_per_sec))
w_sheet.write(r_sheet.nrows+5,4,(lower_right_25 / frame_per_sec))
# DISTANCE
w_sheet.write(r_sheet.nrows+5,5,(dist_lower_right /conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+5,6,(dist_lower_right_75 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+5,7,(dist_lower_right_50 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+5,8,(dist_lower_right_25 / conversion_pixel_to_cm))
# SPEED
speed_lower_right = speed_lower_right_25 = speed_lower_right_50 = speed_lower_right_75 = 0
if lower_right / frame_per_sec > 0:
    speed_lower_right = dist_lower_right/conversion_pixel_to_cm / ( lower_right /frame_per_sec)
if lower_right_25 / frame_per_sec > 0:
    speed_lower_right_25 = dist_lower_right_25/conversion_pixel_to_cm/ (lower_right_25/frame_per_sec)
if lower_right_50 / frame_per_sec > 0:
    speed_lower_right_50 = dist_lower_right_50/conversion_pixel_to_cm/ (lower_right_50/frame_per_sec )
if lower_right_75 / frame_per_sec > 0:
    speed_lower_right_75 = dist_lower_right_75/conversion_pixel_to_cm/ (lower_right_75/ frame_per_sec )
w_sheet.write(r_sheet.nrows+5,9,(speed_lower_right))
w_sheet.write(r_sheet.nrows+5,10,(speed_lower_right_75))
w_sheet.write(r_sheet.nrows+5,11,(speed_lower_right_50))
w_sheet.write(r_sheet.nrows+5,12,(speed_lower_right_25))
# BOUTS
w_sheet.write(r_sheet.nrows+5,13,(lower_right_num_bout))
w_sheet.write(r_sheet.nrows+5,14,(lower_right_frame_bout / frame_per_sec))
w_sheet.write(r_sheet.nrows+5,15,(lower_right_distance_bout/conversion_pixel_to_cm))
for i,f in enumerate(lower_right_lap_bout):
    w_sheet.write(r_sheet.nrows+5,i+16,f)

#print "time_s_upper_right:\t" + str(upper_right/ frame_per_sec)
#print "time_s_upper_right_25:\t" + str(upper_right_25 / frame_per_sec)
#print "time_s_upper_right_50:\t" + str(upper_right_50/ frame_per_sec)
#print "time_s_upper_right_75:\t" + str(upper_right_75/ frame_per_sec)
#
#print "dist_upper_right:\t" + str(dist_upper_right/conversion_pixel_to_cm)
#print "dist_upper_right_25:\t" + str(dist_upper_right_25/conversion_pixel_to_cm)
#print "dist_upper_right_50:\t" + str(dist_upper_right_50/conversion_pixel_to_cm)
#print "dist_upper_right_75:\t" + str(dist_upper_right_75/conversion_pixel_to_cm)
#

#print "speed_upper_right:\t" + str(speed_upper_right)
#print "speed_upper_right_25:\t" + str(speed_upper_right_25)
#print "speed_upper_right_50:\t" + str(speed_upper_right_50)
#print "speed_upper_right_75:\t" + str(speed_upper_right_75)
#
#print "upper_right_num_bout:\t" + str(upper_right_num_bout)
#print "seconds in bout:\t" + str(upper_right_frame_bout / frame_per_sec)
#
#print "upper_right_distance_bout:\t" + str(upper_right_distance_bout/conversion_pixel_to_cm)
#print "upper_right_lap_bout:"
#print "\t".join(map(str,upper_right_lap_bout))


#print "----------------------------"
#print "time_s_upper_left:\t" + str(upper_left/ frame_per_sec)
#print "time_s_upper_left_25:\t" + str(upper_left_25 / frame_per_sec)
#print "time_s_upper_left_50:\t" + str(upper_left_50/ frame_per_sec)
#print "time_s_upper_left_75:\t" + str(upper_left_75/ frame_per_sec)
#
#print "dist_upper_left:\t" + str(dist_upper_left/conversion_pixel_to_cm)
#print "dist_upper_left_25:\t" + str(dist_upper_left_25/conversion_pixel_to_cm)
#print "dist_upper_left_50:\t" + str(dist_upper_left_50/conversion_pixel_to_cm)
#print "dist_upper_left_75:\t" + str(dist_upper_left_75/conversion_pixel_to_cm)
#
#speed_upper_left = speed_upper_left_25 = speed_upper_left_50 = speed_upper_left_75 = 0
#if upper_left / frame_per_sec > 0:
#    speed_upper_left = dist_upper_left/conversion_pixel_to_cm / ( upper_left /frame_per_sec)
#if upper_left_25 / frame_per_sec > 0:
#    speed_upper_left_25 = dist_upper_left_25/conversion_pixel_to_cm/ (upper_left_25/frame_per_sec)
#if upper_left_50 / frame_per_sec > 0:
#    speed_upper_left_50 = dist_upper_left_50/conversion_pixel_to_cm/ (upper_left_50/frame_per_sec )
#if upper_left_75 / frame_per_sec > 0:
#    speed_upper_left_75 = dist_upper_left_75/conversion_pixel_to_cm/ (upper_left_75/ frame_per_sec )
#print "speed_upper_left:\t" + str(speed_upper_left)
#print "speed_upper_left_25:\t" + str(speed_upper_left_25)
#print "speed_upper_left_50:\t" + str(speed_upper_left_50)
#print "speed_upper_left_75:\t" + str(speed_upper_left_75)
#
#print "upper_left_num_bout:\t" + str(upper_left_num_bout)
#print "seconds in bout:\t" + str(upper_left_frame_bout / frame_per_sec)
#
#print "upper_left_distance_bout:\t" + str(upper_left_distance_bout/conversion_pixel_to_cm)
#print "upper_left_lap_bout:"
#print "\t".join(map(str,upper_left_lap_bout))
#
#
#print "----------------------------"
#print "time_s_lower_left:\t" + str(lower_left/ frame_per_sec)
#print "time_s_lower_left_25:\t" + str(lower_left_25 / frame_per_sec)
#print "time_s_lower_left_50:\t" + str(lower_left_50/ frame_per_sec)
#print "time_s_lower_left_75:\t" + str(lower_left_75/ frame_per_sec)
#
#print "dist_lower_left:\t" + str(dist_lower_left/conversion_pixel_to_cm)
#print "dist_lower_left_25:\t" + str(dist_lower_left_25/conversion_pixel_to_cm)
#print "dist_lower_left_50:\t" + str(dist_lower_left_50/conversion_pixel_to_cm)
#print "dist_lower_left_75:\t" + str(dist_lower_left_75/conversion_pixel_to_cm)
#
#speed_lower_left = speed_lower_left_25 = speed_lower_left_50 = speed_lower_left_75 = 0
#if lower_left / frame_per_sec > 0:
#    speed_lower_left = dist_lower_left/conversion_pixel_to_cm / ( lower_left /frame_per_sec)
#if lower_left_25 / frame_per_sec > 0:
#    speed_lower_left_25 = dist_lower_left_25/conversion_pixel_to_cm/ (lower_left_25/frame_per_sec)
#if lower_left_50 / frame_per_sec > 0:
#    speed_lower_left_50 = dist_lower_left_50/conversion_pixel_to_cm/ (lower_left_50/frame_per_sec )
#if lower_left_75 / frame_per_sec > 0:
#    speed_lower_left_75 = dist_lower_left_75/conversion_pixel_to_cm/ (lower_left_75/ frame_per_sec )
#print "speed_lower_left:\t" + str(speed_lower_left)
#print "speed_lower_left_25:\t" + str(speed_lower_left_25)
#print "speed_lower_left_50:\t" + str(speed_lower_left_50)
#print "speed_lower_left_75:\t" + str(speed_lower_left_75)
#
#print "lower_left_num_bout:\t" + str(lower_left_num_bout)
#print "seconds in bout:\t" + str(lower_left_frame_bout / frame_per_sec)
#
#print "lower_left_distance_bout:\t" + str(lower_left_distance_bout/conversion_pixel_to_cm)
#print "lower_left_lap_bout:"
#print "\t".join(map(str,lower_left_lap_bout))
#
#
#print "----------------------------"
#print "time_s_lower_right:\t" + str(lower_right/ frame_per_sec)
#print "time_s_lower_right_25:\t" + str(lower_right_25 / frame_per_sec)
#print "time_s_lower_right_50:\t" + str(lower_right_50/ frame_per_sec)
#print "time_s_lower_right_75:\t" + str(lower_right_75/ frame_per_sec)
#
#print "dist_lower_right:\t" + str(dist_lower_right/conversion_pixel_to_cm)
#print "dist_lower_right_25:\t" + str(dist_lower_right_25/conversion_pixel_to_cm)
#print "dist_lower_right_50:\t" + str(dist_lower_right_50/conversion_pixel_to_cm)
#print "dist_lower_right_75:\t" + str(dist_lower_right_75/conversion_pixel_to_cm)
#
#speed_lower_right = speed_lower_right_25 = speed_lower_right_50 = speed_lower_right_75 = 0
#if lower_right / frame_per_sec > 0:
#    speed_lower_right = dist_lower_right/conversion_pixel_to_cm / ( lower_right /frame_per_sec)
#if lower_right_25 / frame_per_sec > 0:
#    speed_lower_right_25 = dist_lower_right_25/conversion_pixel_to_cm/ (lower_right_25/frame_per_sec)
#if lower_right_50 / frame_per_sec > 0:
#    speed_lower_right_50 = dist_lower_right_50/conversion_pixel_to_cm/ (lower_right_50/frame_per_sec )
#if lower_right_75 / frame_per_sec > 0:
#    speed_lower_right_75 = dist_lower_right_75/conversion_pixel_to_cm/ (lower_right_75/ frame_per_sec )
#print "speed_lower_right:\t" + str(speed_lower_right)
#print "speed_lower_right_25:\t" + str(speed_lower_right_25)
#print "speed_lower_right_50:\t" + str(speed_lower_right_50)
#print "speed_lower_right_75:\t" + str(speed_lower_right_75)
#
#print "lower_right_num_bout:\t" + str(lower_right_num_bout)
#print "seconds in bout:\t" + str(lower_right_frame_bout / frame_per_sec)
#
#print "lower_right_distance_bout:\t" + str(lower_right_distance_bout/conversion_pixel_to_cm)
#print "lower_right_lap_bout:"
#print "\t".join(map(str,lower_right_lap_bout))

wb.save(results_excel)

## Draw heatmap
heatmap, xedges, yedges = np.histogram2d(y, x, bins=50)
extent = [ yedges[0], yedges[-1], xedges[0], xedges[-1]]
plt.clf()
plt.imshow(heatmap, extent=extent)
cb = plt.colorbar()
cb.set_label('mean value')
plt.savefig(sample_name + "_heatmap.png")


