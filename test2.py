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

class ExperimentMetrics(object):
    def __init__(self):
        self.square_25=self.square_50=self.square_75=self.square_100=self.dist_square_25=self.dist_square_50=self.dist_square_75=self.dist_square_100=self.square_num_bout=self.square_frame_bout=self.square_distance_bout=self.counter = 0
        self.last_known_point= [0,0]
        self.square_lap_bout = []
        self.per25_point1 = self.per25_point2 = self.per50_point1 = self.per50_point2 = self.per75_point1 = self.per75_point2 = [0,0]

    # The class "constructor" - It's actually an initializer
    #def __init__():


#upper_left_25=upper_left_50=upper_left_75=upper_left=dist_upper_left_25=dist_upper_left_50=dist_upper_left_75=dist_upper_left=left_known_point=upper_left_num_bout=upper_left_frame_bout=upper_left_distance_bout=left_counter = 0
#upper_left_lap_bout = []
#upper_right_25=upper_right_50=upper_right_75=upper_right=dist_upper_right_25=dist_upper_right_50=dist_upper_right_75=dist_upper_right=right_known_point=upper_right_num_bout=upper_right_frame_bout=upper_right_distance_bout=right_counter = 0
#upper_right_lap_bout = []

if len(sys.argv) != 4:
    sys.stderr.write("usage: python test.py video_file results.xls dark/bright/medium\n")
    sys.exit()
# create video capture
#cap = cv2.VideoCapture("/Users/hayer/Desktop/Anand/openfields/100611_openfield-b5.m4v")
#cap = cv2.VideoCapture("/Users/kat/Desktop/071411_batch4-openfield.m4v")
cap = cv2.VideoCapture(sys.argv[1])
sample_name = sys.argv[1].split("/")[-1].split(".")[0]
global brightness
brightness = sys.argv[3]


print sample_name


##if (!cap.isOpened()):  // check if we succeeded
 ##   return -1;
print cap.isOpened()

total_number_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT)
print "Total number of frames"
print total_number_of_frames
frame_per_sec = cap.get(CV_CAP_PROP_FPS)
##frame_per_sec = 30.03888889
print "Frame per sec"
print frame_per_sec
increaser = int(frame_per_sec/5)

def get_threshold(imgray,brightness):
    if brightness == "bright":
        #thresh = cv2.inRange(imgray,np.array((200)), np.array((300)))
        thresh = cv2.inRange(imgray,np.array((180,180,180)), np.array((300, 300, 300)))
    elif brightness == "medium":
        #thresh = cv2.inRange(imgray,np.array((200,200,200)), np.array((250,250,250)))
        thresh = cv2.inRange(imgray,np.array((180,180,180)), np.array((250,250,250)))
    elif brightness == "mediumlow":
        thresh = cv2.inRange(imgray,np.array((160,160,160)), np.array((180,180,180)))
    elif brightness == "brighter":
        thresh = cv2.inRange(imgray,np.array((250,250,250)), np.array((300, 300, 300)))
    elif brightness == "dark":
        thresh = cv2.inRange(imgray,np.array((90,90,90)), np.array((110, 110, 110)))
    elif brightness == "dark2":
        thresh = cv2.inRange(imgray,np.array((120, 120, 120)), np.array((160,160,160)))
    elif brightness == "dark2.2":
        thresh = cv2.inRange(imgray,np.array((130, 130, 130)), np.array((170,170,170)))
    elif brightness == "dark2.3":
        thresh = cv2.inRange(imgray,np.array((130, 130, 130)), np.array((180,180,180)))
    elif brightness == "dark2.4":
        thresh = cv2.inRange(imgray,np.array((125, 125, 125)), np.array((180,180,180)))
    elif brightness == "dark3":
        thresh = cv2.inRange(imgray,np.array((150,150,150)), np.array((180,180,180)))
    elif brightness == "dark4":
        thresh = cv2.inRange(imgray,np.array((157,157,157)), np.array((220,220,220)))
    else:
        thresh = cv2.inRange(imgray,np.array((100,100,100)), np.array((160, 160, 160)))
    return thresh

def draw_rect(pic,height,width,percentage,adjust=0):
    w = int((width)*percentage)
    x = adjust+int((width-width*percentage)/2)
    h = int(height*percentage)
    y = int((height-height*percentage)/2)
    cv2.rectangle(pic,(x,y),(x+w,y+h),(0,255,0),1)
    return


def comp_tuple(mp,pt1,pt2):
    return mp[0] >= pt1[0] and mp[0] <= pt2[0] and mp[1] >= pt1[1] and mp[1] <= pt2[1]

def dist(pt1,pt2):
    xd = pt2[0] - pt1[0]
    yd = pt2[1] - pt1[1]
    return math.sqrt(xd*xd + yd*yd)


def analyze(frame,frame_name,frame2,x,y,exp_obj,width):
    # smooth it
    #cv2.imshow('before',frame)
    #width = frame2.get(CV_CAP_PROP_FRAME_WIDTH)
    frame3 = cv2.blur(frame,(17,17))
    #cv2.imshow('blur',frame)
    # convert to hsv and find range of colors
    hsv = frame3 #cv2.cvtColor(frame3,cv2.COLOR_BGR2HSV)
    if brightness == "bright":
        thresh = cv2.inRange(hsv,np.array((1, 1, 1)), np.array((80, 80, 80)))
    elif brightness == "medium":
        #thresh = cv2.inRange(hsv,np.array((1, 1, 1)), np.array((90, 90, 90)))
        thresh = cv2.inRange(hsv,np.array((1,1, 1)), np.array((60, 60, 60)))
    elif brightness == "mediumlow":
        #thresh = cv2.inRange(hsv,np.array((1, 1, 1)), np.array((90, 90, 90)))
        thresh = cv2.inRange(hsv,np.array((1,1, 1)), np.array((10, 10, 10)))
    elif brightness == "brighter":
        thresh = cv2.inRange(hsv,np.array((1, 1, 1)), np.array((150, 150, 150)))
    elif brightness == "dark":
        thresh = cv2.inRange(hsv,np.array((5,5, 5)), np.array((10, 10, 10)))
    elif brightness == "dark2":
        thresh = cv2.inRange(hsv,np.array((1,1, 1)), np.array((10, 10, 10)))
    elif brightness == "dark2.2":
        thresh = cv2.inRange(hsv,np.array((2,2, 2)), np.array((20, 20, 20)))
    elif brightness == "dark2.3":
        thresh = cv2.inRange(hsv,np.array((4,4,4)), np.array((20, 20, 20)))
    elif brightness == "dark3":
        thresh = cv2.inRange(hsv,np.array((0,0, 0)), np.array((10, 10, 10)))
    elif brightness == "dark4":
        thresh = cv2.inRange(hsv,np.array((0,0, 0)), np.array((5, 5, 5)))
    else:
        thresh = cv2.inRange(hsv,np.array((0,0, 0)), np.array((15, 15, 15)))

    #res = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,element)
    #cv2.imshow('thresh_first2',res)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
    cv2.erode(thresh,element,thresh,None,2)
    cv2.dilate(thresh,element,thresh,None,10)
    #cv2.imshow('thresh_first',thresh)
    #element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))

    #cv2.imshow('thresh_2',thresh)
    #cv2.dilate(thresh,element,thresh,None,10)
    thresh2 = thresh.copy()

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    draw_rect(frame,height,width,0.7)
    draw_rect(frame,height,width,0.45)
    draw_rect(frame,height,width,0.2)
    #x = exp_obj.per25_point1[0]
    #y = exp_obj.per25_point1[1]
    cv2.rectangle(frame,tuple(exp_obj.per25_point1),tuple(exp_obj.per25_point2),(255,0,0),1)
    #cv2.imshow(frame_name,frame)

    # finding contour with maximum area and store it as best_cnt
    max_area = 0
    contours2 = contours
    i = 0
    best_cnt = []
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
    #start = tuple([0,height/2])
    #end = tuple([width,height/2])
    #cv2.line(frame,start,end,cv.CV_RGB(255,0,255))
    #start = tuple([width,0])
    #end = tuple([width/2,height])
    #cv2.line(frame,start,end,cv.CV_RGB(255,0,255))
    #cv2.line(frame,start,end,cv.CV_RGB(255,0,255))

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
        #x.append(mid_point[0])
        #y.append(mid_point[1])
        mid_point = tuple([int(mid_point[0]),int(mid_point[1])])
        mid_points.append(mid_point)
        #cv2.circle(frame2,mid_point, 1, cv.CV_RGB(255,0,0))
        #cv2.circle(frame,mid_point, 1, cv.CV_RGB(255,0,0))

    closest = None
    distance = 3000
    for mp in mid_points:
        if distance > dist(exp_obj.last_known_point,mp):
            distance = dist(exp_obj.last_known_point,mp)
            closest = mp


    in_mid_points = 0
    if closest is not None:
        mp = closest

        x.append(mp[0])
        y.append(mp[1])
        cv2.circle(frame2,mp, 1, cv.CV_RGB(255,0,0))
        cv2.circle(frame,mp, 1, cv.CV_RGB(255,0,0))
        in_mid_points = 1
        exp_obj.last_known_point = exp_obj.last_known_point or mp
        distance = dist(exp_obj.last_known_point,mp)
        if distance > dist_threshold:
            exp_obj.counter = exp_obj.counter + 1
            if exp_obj.counter == bout_threshold:
                exp_obj.square_num_bout = exp_obj.square_num_bout + 1
                exp_obj.square_lap_bout.append(0)
            if exp_obj.counter >= bout_threshold:
                cv2.circle(frame,mp, 5, cv.CV_RGB(255,0,255))
                exp_obj.square_frame_bout = exp_obj.square_frame_bout + 1
                exp_obj.square_distance_bout = exp_obj.square_distance_bout + distance
                exp_obj.square_lap_bout[-1] = exp_obj.square_lap_bout[-1] +1
        else:
            distance = 0
            exp_obj.counter = 0

        if comp_tuple(mp,exp_obj.per25_point1,exp_obj.per25_point2):
            exp_obj.dist_square_25 = exp_obj.dist_square_25 + distance
            exp_obj.square_25 = exp_obj.square_25 + 1
        elif comp_tuple(mp,exp_obj.per50_point1,exp_obj.per50_point2):
            exp_obj.square_50 = exp_obj.square_50 + 1
            exp_obj.dist_square_50 = exp_obj.dist_square_50 + distance
        elif comp_tuple(mp,exp_obj.per75_point1,exp_obj.per75_point2):
            exp_obj.square_75 = exp_obj.square_75 + 1
            exp_obj.dist_square_75 = exp_obj.dist_square_75 + distance
        else:
            exp_obj.square_100 = exp_obj.square_100 + 1
            exp_obj.dist_square_100 = exp_obj.dist_square_100 + distance
        exp_obj.last_known_point = mp
        #last_known_point = mp

    if in_mid_points == 0:
        mp = exp_obj.last_known_point
        if comp_tuple(mp,exp_obj.per25_point1,exp_obj.per25_point2):
            exp_obj.square_25 = exp_obj.square_25 + 1
        elif comp_tuple(mp,exp_obj.per50_point1,exp_obj.per50_point2):
            exp_obj.square_50 = exp_obj.square_50 + 1
        elif comp_tuple(mp,exp_obj.per75_point1,exp_obj.per75_point2):
            exp_obj.square_75 = exp_obj.square_75 + 1
        else:
            exp_obj.square_100 = exp_obj.square_100 + 1

    # finding centroids of best_cnt and draw a circle there
    #if not best_cnt:
    #    print "List is empty!"
    #else:
    #    M = cv2.moments(best_cnt)
    #    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    #    cv2.circle(frame,(cx,cy),5,255,-1)

    # Show it, if key pressed is 'Esc', exit the loop

    #cv2.imshow('%sthresh' % (frame_name),thresh2)
    #cv2.imshow('%scontour' % (frame_name),frame2)
    cv2.imshow('%sframe' % (frame_name),frame)
    return

cap.set(CV_CAP_PROP_POS_FRAMES,int(frame_per_sec*30))
_,frame2 = cap.read()
imgray = cv2.blur(frame2,(15,15))
#imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)

thresh = get_threshold(imgray,brightness)
    #thresh = cv2.inRange(imgray,np.array((150,150,150)), np.array((160, 160, 160)))
#element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(5,5))
#cv2.erode(thresh,element,thresh,None,6)
#cv2.dilate(thresh,element,thresh,None,10)
cv2.imwrite("gray_tra1.png",thresh)

cv2.imshow('thresh',thresh)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
num_p = 0
all_x = []
all_y = []
actual_width = int(cap.get(3))
actual_height = int(cap.get(4))
print "Actual height"
print actual_height
print "actual_width"
print actual_width
while not contours:
    _,frame2 = cap.read()
    cv2.imshow('frame2',frame2)

    imgray = cv2.blur(frame2,(15,15))
    #imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2HSV)

    thresh = get_threshold(imgray,brightness)
    #thresh = cv2.inRange(imgray,np.array((110,110,110)), np.array((300, 300, 300)))
    #imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgray',imgray)
    #ret,thresh = cv2.threshold(imgray,170,200,0)
    #cv2.imwrite("gray_tra.png",thresh)
    #cv2.imshow('thresh',thresh)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("gray_tra.png",thresh)

for cnt in contours:
    for point in cnt:
        num_p = num_p + 1
        x = point[0][0]
        y = point[0][1]
        if x > (actual_width/2)-(actual_width/2)*0.7 and x<(actual_width/2)+(actual_width/2)*0.7 and y > (actual_height/2)-(actual_height/2)*0.7 and y < (actual_height/2)+(actual_height/2)*0.7 : # and y > 20 and y < 460:
            all_x.append(x)
            all_y.append(y)
print (all_x)
print (all_y)
print np.mean(all_x)
print np.mean(all_y)





width = 2*int(np.mean(all_x))
height = 2*int(np.mean(all_y))

diff_width = actual_width - width
diff_height = actual_height - height

cv_rect_obj = frame2[0:height,0:width]
frame2 = frame2[0:height,0:width]

#cv_rect_obj = cv2.cvtColor(cv_rect_obj,cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_test.png",cv_rect_obj)
imgray = cv2.blur(cv_rect_obj,(15,15))

thresh = get_threshold(imgray,brightness)
cv2.imwrite("gray_test3.png",thresh)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
#cv2.erode(thresh,element,thresh,None,10)
cv2.dilate(thresh,element,thresh,None,10)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(cv_rect_obj,contours,-1,cv.CV_RGB(255,255,0),1)

areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

x_in,y_in,w_in,h_in = cv2.boundingRect(cnt)
#cv2.rectangle(cv_rect_obj,(x,y),(x+w,y+h),(0,255,0),2)
cv_rect_obj = cv_rect_obj[y_in:y_in+h_in,x_in:w_in+x_in]
frame2 = frame2[y_in:y_in+h_in,x_in:w_in+x_in]
#cv2.imshow('thresh_first',thresh)
#element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
imgray = cv2.blur(cv_rect_obj,(15,15))
#imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,170,200,0)

tresh = get_threshold(imgray,brightness)
#thresh = cv2.inRange(imgray,np.array((110,110,110)), np.array((300, 300, 300)))
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
#cv2.erode(thresh,element,thresh,None,6)
#cv2.dilate(thresh,element,thresh,None,6)
cv2.imwrite("gray_tra2.png",thresh)
cv2.imshow('thresh',thresh)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
num_p = 0
all_x = []
all_y = []

for cnt in contours:
    for point in cnt:
        num_p = num_p + 1
        x = point[0][0]
        y = point[0][1]
        if x > (w_in/2)-(w_in/2)*0.7 and x<(w_in/2)+(w_in/2)*0.7 and y > (h_in/2)-(h_in/2)*0.7 and y < (h_in/2)+(h_in/2)*0.7 : # and y > 20 and y < 460:
            all_x.append(x)
            all_y.append(y)

width = int(np.mean(all_x))
height = h_in

width = w_in/2




cv_rect_obj1 = cv_rect_obj[0:height,0:width]
#draw_rect(cv_rect_obj1,height,width,0.7)
#draw_rect(cv_rect_obj1,height,width,0.45)
#draw_rect(cv_rect_obj1,height,width,0.2)
#cv2.imwrite("gray_test3.png",cv_rect_obj1)
cv_rect_obj2 = cv_rect_obj[0:height,width:width*2]
#draw_rect(cv_rect_obj2,height,width,0.7)
#draw_rect(cv_rect_obj2,height,width,0.45)
#draw_rect(cv_rect_obj2,height,width,0.2)
#cv2.imwrite("gray_test2.png",cv_rect_obj2)




#half_width = int(width/2)
#half_height = int(height/2)
#
#perc_75_width = int((half_width-half_width*0.55)/2)
#perc_75_height = int((height-height*0.55)/2)
#perc_50_width = int((half_width-half_width*0.35)/2)
#perc_50_height = int((height-height*0.35)/2)
#perc_25_width = int((half_width-half_width*0.15)/2)
#perc_25_height = int((height-height*0.15)/2)

bout_threshold = 8
dist_threshold = 8
conversion_pixel_to_cm = 15

## Points for heatmap
x_left = []
y_left = []
mid_points = []

x_right = []
y_right = []

#upper_left_25=upper_left_50=upper_left_75=upper_left=dist_upper_left_25=dist_upper_left_50=dist_upper_left_75=dist_upper_left=left_known_point=upper_left_num_bout=upper_left_frame_bout=upper_left_distance_bout=left_counter = 0
#upper_left_lap_bout = []
#upper_right_25=upper_right_50=upper_right_75=upper_right=dist_upper_right_25=dist_upper_right_50=dist_upper_right_75=dist_upper_right=right_known_point=upper_right_num_bout=upper_right_frame_bout=upper_right_distance_bout=right_counter = 0
#upper_right_lap_bout = []



time_in_msec = 0
#frame_per_sec = 0
left_known_point = right_known_point = [0,0]

left_object = ExperimentMetrics()
right_object = ExperimentMetrics()

width_l = w_in/2
width_r = w_in/2

left_object.per75_point1 = [int(width_l-width_l*0.7)/2,int((height-height*0.7)/2)]
left_object.per75_point2 = [int((width_l-width_l*0.7)/2+width_l*0.7),int((height-height*0.7)/2+height*0.7)]
left_object.per50_point1 = [int(width_l-width_l*0.45)/2,int((height-height*0.45)/2)]
left_object.per50_point2 = [int((width_l-width_l*0.45)/2+width_l*0.45),int((height-height*0.45)/2+height*0.45)]
left_object.per25_point1 = [int(width_l-width_l*0.2)/2,int((height-height*0.2)/2)]
left_object.per25_point2 = [int((width_l-width_l*0.2)/2+width_l*0.2),int((height-height*0.2)/2+height*0.2)]
right_object.per75_point1 = [int(width_r-width_r*0.7)/2,int((height-height*0.7)/2)]
right_object.per75_point2 = [int((width_r-width_r*0.7)/2+width_r*0.7),int((height-height*0.7)/2+height*0.7)]
right_object.per50_point1 = [int(width_r-width_r*0.45)/2,int((height-height*0.45)/2)]
right_object.per50_point2 = [int((width_r-width_r*0.45)/2+width_r*0.45),int((height-height*0.45)/2+height*0.45)]
right_object.per25_point1 = [int(width_r-width_r*0.2)/2,int((height-height*0.2)/2)]
right_object.per25_point2 = [int((width_r-width_r*0.2)/2+width_r*0.2),int((height-height*0.2)/2+height*0.2)]

cap.set(CV_CAP_PROP_POS_FRAMES,0)
frame_number = cap.get(CV_CAP_PROP_POS_FRAMES)



while(frame_number < total_number_of_frames):
#while(frame_number < 450):
    #frame_number = cap.get(CV_CAP_PROP_POS_FRAMES)
    #CV_CAP_PROP_POS_MSEC
    #l = cap.get(CV_CAP_PROP_POS_MSEC)
    #if l <= 1000:
    #    print l
    #    frame_per_sec += 1.0
    # read the frames

    cap.set(CV_CAP_PROP_POS_FRAMES,frame_number)
    _,frame = cap.read()
    frame = frame[y_in:y_in+h_in,x_in:w_in+x_in]
    if cap.get(CV_CAP_PROP_POS_MSEC) > 0.0:
        time_in_msec = cap.get(CV_CAP_PROP_POS_MSEC)
    percent = frame_number/total_number_of_frames * 100
    l = int(percent/2)
    if l%2==0:
      sys.stderr.write("\r[%-50s] %d%%" % ('='*int(l), percent))
      sys.stderr.flush()
    #capture = cv.CaptureFromFile("/Users/hayer/Desktop/Anand/openfields/071211_Batch1-openfield.m4v")

    left_side = frame[0:height,0:width]
    right_side = frame[0:height,width:width*2]
    #upper_left_25 = upper_left_25 +1
    analyze(left_side,"left",cv_rect_obj1,x_left,y_left,left_object,width_l)
    #print left_object.square_50
    analyze(right_side,"right",cv_rect_obj2,x_right,y_right,right_object,width_r)
    #print right_object.square_50
    if cv2.waitKey(1) == 27:
        break

    frame_number = int(frame_number+increaser)
    #print left_object.square_lap_bout
    #print right_object.square_lap_bout
draw_rect(frame2,height,width,0.7)
draw_rect(frame2,height,width,0.45)
draw_rect(frame2,height,width,0.2)

draw_rect(frame2,height,width,0.7,width)
draw_rect(frame2,height,width,0.45,width)
draw_rect(frame2,height,width,0.2,width)
cv2.imwrite(sample_name + "_tra.png",frame2)
# Clean up everything before leaving
cv2.destroyAllWindows()

print "ENDE"
#print left_object.square_50
#print right_object.square_50
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
# Video information
w_sheet.write(r_sheet.nrows+1,5,"total_number_of_frames: ")
w_sheet.write(r_sheet.nrows+1,6,total_number_of_frames)
w_sheet.write(r_sheet.nrows+1,7,"Frames per second: ")
w_sheet.write(r_sheet.nrows+1,8,frame_per_sec)
### UPPER LEFT
w_sheet.write(r_sheet.nrows+2,0,"Left")
# FIXTIME:
left_object.square_100 = left_object.square_100 * increaser
left_object.square_75 = left_object.square_75 * increaser
left_object.square_50 = left_object.square_50 * increaser
left_object.square_25 = left_object.square_25 * increaser

# TIME
w_sheet.write(r_sheet.nrows+2,1,(left_object.square_100    / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,2,(left_object.square_75 / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,3,(left_object.square_50 / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,4,(left_object.square_25 / frame_per_sec))
# DISTANCE
w_sheet.write(r_sheet.nrows+2,5,(left_object.dist_square_100 /conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+2,6,(left_object.dist_square_75 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+2,7,(left_object.dist_square_50 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+2,8,(left_object.dist_square_25 / conversion_pixel_to_cm))
# SPEED
speed_upper_left = speed_upper_left_25 = speed_upper_left_50 = speed_upper_left_75 = 0
if left_object.square_100 / frame_per_sec > 0:
    speed_upper_left = left_object.dist_square_100 /conversion_pixel_to_cm / ( left_object.square_100 /frame_per_sec)
if left_object.square_25 / frame_per_sec > 0:
    speed_upper_left_25 = left_object.dist_square_25 /conversion_pixel_to_cm/ (left_object.square_25/frame_per_sec)
if left_object.square_50 / frame_per_sec > 0:
    speed_upper_left_50 = left_object.dist_square_50 /conversion_pixel_to_cm/ (left_object.square_50/frame_per_sec )
if left_object.square_75 / frame_per_sec > 0:
    speed_upper_left_75 = left_object.dist_square_75 /conversion_pixel_to_cm/ (left_object.square_75/ frame_per_sec )
w_sheet.write(r_sheet.nrows+2,9,(speed_upper_left))
w_sheet.write(r_sheet.nrows+2,10,(speed_upper_left_75))
w_sheet.write(r_sheet.nrows+2,11,(speed_upper_left_50))
w_sheet.write(r_sheet.nrows+2,12,(speed_upper_left_25))
# BOUTS
w_sheet.write(r_sheet.nrows+2,13,(left_object.square_num_bout))
w_sheet.write(r_sheet.nrows+2,14,(left_object.square_frame_bout*increaser / frame_per_sec))
w_sheet.write(r_sheet.nrows+2,15,(left_object.square_distance_bout/conversion_pixel_to_cm))
for i,f in enumerate(left_object.square_lap_bout):
    w_sheet.write(r_sheet.nrows+2,i+16,f*increaser/ frame_per_sec)

### UPPER RIGHT
w_sheet.write(r_sheet.nrows+3,0,"Right")
# FIXTIME:
right_object.square_100 = right_object.square_100 * increaser
right_object.square_75 = right_object.square_75 * increaser
right_object.square_50 = right_object.square_50 * increaser
right_object.square_25 = right_object.square_25 * increaser

# TIME
w_sheet.write(r_sheet.nrows+3,1,(right_object.square_100    / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,2,(right_object.square_75 / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,3,(right_object.square_50 / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,4,(right_object.square_25 / frame_per_sec))
# DISTANCE
w_sheet.write(r_sheet.nrows+3,5,(right_object.dist_square_100 /conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+3,6,(right_object.dist_square_75 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+3,7,(right_object.dist_square_50 / conversion_pixel_to_cm))
w_sheet.write(r_sheet.nrows+3,8,(right_object.dist_square_25 / conversion_pixel_to_cm))
# SPEED
speed_upper_right = speed_upper_right_25 = speed_upper_right_50 = speed_upper_right_75 = 0
if right_object.square_100 / frame_per_sec > 0:
    speed_upper_right = right_object.dist_square_100 /conversion_pixel_to_cm / ( right_object.square_100 /frame_per_sec)
if right_object.square_25 / frame_per_sec > 0:
    speed_upper_right_25 = right_object.dist_square_25 /conversion_pixel_to_cm/ (right_object.square_25/frame_per_sec)
if right_object.square_50 / frame_per_sec > 0:
    speed_upper_right_50 = right_object.dist_square_50 /conversion_pixel_to_cm/ (right_object.square_50/frame_per_sec )
if right_object.square_75 / frame_per_sec > 0:
    speed_upper_right_75 = right_object.dist_square_75 /conversion_pixel_to_cm/ (right_object.square_75/ frame_per_sec )
w_sheet.write(r_sheet.nrows+3,9,(speed_upper_right))
w_sheet.write(r_sheet.nrows+3,10,(speed_upper_right_75))
w_sheet.write(r_sheet.nrows+3,11,(speed_upper_right_50))
w_sheet.write(r_sheet.nrows+3,12,(speed_upper_right_25))
# BOUTS
w_sheet.write(r_sheet.nrows+3,13,(right_object.square_num_bout))
w_sheet.write(r_sheet.nrows+3,14,(right_object.square_frame_bout*increaser / frame_per_sec))
w_sheet.write(r_sheet.nrows+3,15,(right_object.square_distance_bout/conversion_pixel_to_cm))
for i,f in enumerate(right_object.square_lap_bout):
    w_sheet.write(r_sheet.nrows+3,i+16,f*increaser/ frame_per_sec)




wb.save(results_excel)

## Draw heatmap
if y_left and x_left:
    heatmap, xedges, yedges = np.histogram2d(y_left, x_left, bins=50)
    extent = [ yedges[0], yedges[-1], xedges[0], xedges[-1]]
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    cb = plt.colorbar()
    cb.set_label('mean value')
    plt.savefig(sample_name + "_left_heatmap.png")


if y_right and x_right:
    heatmap, xedges, yedges = np.histogram2d(y_right, x_right, bins=50)
    extent = [ yedges[0], yedges[-1], xedges[0], xedges[-1]]
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    cb = plt.colorbar()
    cb.set_label('mean value')
    plt.savefig(sample_name + "_right_heatmap.png")


