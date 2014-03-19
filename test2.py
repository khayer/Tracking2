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


def analyze(frame,frame_name,frame2):
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


    draw_rect(frame,height,width/2,0.7)
    draw_rect(frame,height,width/2,0.45)
    draw_rect(frame,height,width/2,0.2)
    cv2.imshow(frame_name,frame)

    max_area = 0
    contours2 = contours

    cv2.drawContours(frame,contours2,-1,cv.CV_RGB(255,255,0),1)

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

    cv2.imshow('%sthresh' % (frame_name),thresh2)
    cv2.imshow('%scontour' % (frame_name),frame2)
    cv2.imshow('%sframe' % (frame_name),frame)
    

_,frame2 = cap.read()
imgray = cv2.blur(frame2,(15,15))
imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,170,200,0)
cv2.imwrite("gray_tra.png",thresh)
cv2.imshow('thresh',thresh)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
num_p = 0
all_x = []
all_y = []
while not contours:
    _,frame2 = cap.read()
    cv2.imshow('frame2',frame2)
    imgray = cv2.blur(frame2,(15,15))
    imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('imgray',imgray)
    ret,thresh = cv2.threshold(imgray,170,200,0)
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

cv_rect_obj = frame2[0:height,0:width]
frame2= frame2[0:height,0:width]
cv2.imwrite("gray_test.png",cv_rect_obj)
#hsv = cv2.cvtColor(cv_rect_obj,cv2.COLOR_BGR2HSV)
imgray = cv2.blur(cv_rect_obj,(15,15))

thresh = cv2.inRange(imgray,np.array((90, 90, 90)), np.array((160, 160, 160)))
cv2.imwrite("gray_test3.png",thresh)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,3))
cv2.erode(thresh,element,thresh,None,2)
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
imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,170,200,0)
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
        if x > 20 and x < 640 and y > 20 and y < 460:
            all_x.append(x)
            all_y.append(y)

width = 2*int(np.median(all_x))
height = 2*int(np.median(all_y))

def draw_rect(pic,height,width,percentage):
    w = int(width*percentage)
    x = int((width-width*percentage)/2)
    h = int(height*percentage)
    y = int((height-height*percentage)/2)
    cv2.rectangle(pic,(x,y),(x+w,y+h),(0,255,0),1)
    return

cv_rect_obj1 = frame2[0:height,0:width/2]
cv_rect_obj2 = frame2[0:height,width/2:width]

def comp_tuple(mp,pt1,pt2):
    return mp[0] >= pt1[0] and mp[0] <= pt2[0] and mp[1] >= pt1[1] and mp[1] <= pt2[1]

def dist(pt1,pt2):
    xd = pt2[0] - pt1[0]
    yd = pt2[1] - pt1[1]
    return math.sqrt(xd*xd + yd*yd)

#half_width = int(width/2)
#half_height = int(height/2)
#
#perc_75_width = int((half_width-half_width*0.55)/2)
#perc_75_height = int((height-height*0.55)/2)
#perc_50_width = int((half_width-half_width*0.35)/2)
#perc_50_height = int((height-height*0.35)/2)
#perc_25_width = int((half_width-half_width*0.15)/2)
#perc_25_height = int((height-height*0.15)/2)

bout_threshold = 11
dist_threshold = 3
conversion_pixel_to_cm = 10

## Points for heatmap
x = []
y = []
mid_points = []

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
    frame_number = int(frame_number+10)
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

    left_side = frame[0:height,0:width/2]
    right_side = frame[0:height,width/2:width]

    analyze(left_side,"left",cv_rect_obj1)
    analyze(right_side,"right",cv_rect_obj2)

    if cv2.waitKey(33) == 27:
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
w_sheet.write(r_sheet.nrows+3,0,"right")
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


wb.save(results_excel)

## Draw heatmap
heatmap, xedges, yedges = np.histogram2d(y, x, bins=50)
extent = [ yedges[0], yedges[-1], xedges[0], xedges[-1]]
plt.clf()
plt.imshow(heatmap, extent=extent)
cb = plt.colorbar()
cb.set_label('mean value')
plt.savefig(sample_name + "_heatmap.png")
