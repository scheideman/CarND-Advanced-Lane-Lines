import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

class Line():
    
    def __init__(self):
        self.n = 15
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def get_line(linex_base,img,margin, minpix,old_line):

    out_img = np.dstack((img, img, img))*255
    line = old_line
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    x_current = linex_base

    lane_inds = []

    countGoodWindows = 0
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
            countGoodWindows += 1
    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)
    
    linex = nonzerox[lane_inds]
    liney = nonzeroy[lane_inds] 
    
    # Fit a second order polynomial to each
    line_fit = np.polyfit(liney, linex, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    if(len(line.recent_xfitted) > line.n):
        line.recent_xfitted.pop(0)

    line.recent_xfitted.append(fitx)
    line.bestx = np.mean(line.recent_xfitted,axis=0)
    line.best_fit = np.polyfit(ploty, line.bestx, 2)
    line.current_fit = line_fit
    line.radius_of_curvature = get_radius_curvature(line.best_fit)
    line.line_base_pos = abs(img.shape[1] / 2 - fitx[-1]) * xm_per_pix 
    line.allx = linex
    line.ally = liney

    out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]

    #cv2.imwrite("output_images/"+"lane_line2_"+str(line.radius_of_curvature)+".jpg",out_img)
    return line

def find_lane_lines(img,right_line,left_line):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    out_img = np.dstack((img, img, img))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    lane_width_pixels = rightx_base-leftx_base

    if(sanity_check(img) is False):
        return False, right_line, left_line
    
    new_right_line = get_line(rightx_base,img,100, 50, right_line)
    new_left_line = get_line(leftx_base,img,100, 50, left_line)
    
    if(abs(right_line.radius_of_curvature - left_line.radius_of_curvature)>10000):
        return False, right_line, left_line

    return True, new_left_line, new_right_line


#
# Function to check if the lane lines found in the binary image make sense
#
def sanity_check(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    out_img = np.dstack((img, img, img))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    histogram_top = np.sum(img[:img.shape[0]/2,:], axis=0)
    leftx_base_top = np.argmax(histogram_top[:midpoint])
    rightx_base_top = np.argmax(histogram_top[midpoint:]) + midpoint


    lane_width_pixels = rightx_base-leftx_base
    lane_width_pixels_top = rightx_base_top-leftx_base_top

    if(leftx_base <= 45 or rightx_base <= 45):
        #bad binary image
        return False
    if(abs(lane_width_pixels - lane_width_pixels_top) > 150):
        # lanes not parallel
        return False
    if(lane_width_pixels < 500 or lane_width_pixels > 750):
        # lane not the right width 
        return False

    return True


def update_lane_lines_with_new_frame(img,left_line, right_line):

    if(sanity_check(img) is False):
        return False, left_line, right_line
    
    left_line = update_line(img,left_line)
    right_line = update_line(img,right_line)
    #print("Update!!")
    return True,left_line, right_line

#
# Update line model using posterior line model
#
def update_line(img,line):
    line_fit = line.current_fit

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    line_lane_inds = ((nonzerox > (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] - margin)) & (nonzerox < (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] + margin))) 

    # Again, extract left and right line pixel positions
    linex = nonzerox[line_lane_inds]
    liney = nonzeroy[line_lane_inds] 
    # Fit a second order polynomial to each
    line_fit = np.polyfit(liney, linex, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img))*255
   
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[line_lane_inds], nonzerox[line_lane_inds]] = [255, 0, 0]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    line_line_window1 = np.array([np.transpose(np.vstack([line_fitx-margin, ploty]))])
    line_line_window2 = np.array([np.flipud(np.transpose(np.vstack([line_fitx+margin, ploty])))])
    line_line_pts = np.hstack((line_line_window1, line_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([line_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    if(len(line.recent_xfitted) > line.n):
        line.recent_xfitted.pop(0)

    line.recent_xfitted.append(line_fitx)
    line.bestx = np.mean(line.recent_xfitted,axis=0)
    line.best_fit = np.polyfit(ploty, line.bestx, 2)
    line.current_fit = line_fit
    line.radius_of_curvature = get_radius_curvature(line.best_fit)
    line.line_base_pos = abs(img.shape[1] / 2 - line_fitx[-1]) * xm_per_pix 
    line.allx = linex
    line.ally = liney

    return line


#
# Get the radius of curvature for a polynomial line 
#
def get_radius_curvature(line_fit):
    ploty = np.linspace(0, 719, num=720)

    #bottom of image, ie where the car is 
    y_eval = np.max(ploty)

    plotx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    fit_world = np.polyfit(ploty*ym_per_pix, plotx*xm_per_pix, 2)
    
    curverad = ((1 + (2*fit_world[0]*y_eval*ym_per_pix + fit_world[1])**2)**1.5) / np.absolute(2*fit_world[0])
    return curverad