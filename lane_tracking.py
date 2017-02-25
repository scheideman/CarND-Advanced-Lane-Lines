import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.n = 10
        # was the line detected in the last iteration?
        self.detected = False  
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
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
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
        #cv2.imshow("out_img",out_img)
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

    
    line.diffs = np.subtract(line_fit,line.current_fit)
    line.recent_xfitted.append(fitx)
    line.bestx = np.mean(line.recent_xfitted,axis=0)
    line.best_fit = np.polyfit(liney, line.bestx, 2)
    line.current_fit = line_fit
    line.radius_of_curvature = get_radius_curvature(best_fit)
    line.line_base_pos = abs(img.shape[1] / 2 - linex_base) * xm_per_pix 
    line.allx = linex
    line.ally = liney


    out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
    
    return line_fit

def find_lane_lines(img):
    right_line = Line()
    left_line = Line()

    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    out_img = np.dstack((img, img, img))*255

    midpoint = np.int(histogram.shape[0]/2)
    #plt.plot(histogram)
    #plt.waitforbuttonpress()
    #plt.close()
    # TODO check if max is greater than some threshold to be a line
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    print("diffleft: ",leftx_base)
    print("diffright:", rightx_base)
    lane_width_pixels = rightx_base-leftx_base

    if(sanity_check(img) is False):
        return False, None, None
    

    center_offset = (img.shape[1]/2) - ((rightx_base - leftx_base)/2 + leftx_base)
    print("offset: " , center_offset)
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        #cv2.imshow("out_img",out_img)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.waitforbuttonpress()
    #plt.close()

    
    
    return True, left_fit, right_fit

def sanity_check(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    out_img = np.dstack((img, img, img))*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    lane_width_pixels = rightx_base-leftx_base

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


def update_line_with_new_frame(img,left_fit, right_fit):

    if(sanity_check(img) is False):
        return False, None, None

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    #plt.imshow(result)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)

    #plt.waitforbuttonpress()
    #plt.close()
    

    return left_fit, right_fit

def update_line(img,line, linex_base):

    if(sanity_check(img) is False):
        return False, None, None

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
    #plt.imshow(result)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)

    #plt.waitforbuttonpress()
    #plt.close()
    if(len(line.recent_xfitted) > line.n):
        line.recent_xfitted.pop(0)

    line.diffs = np.subtract(line_fit,line.current_fit)
    line.recent_xfitted.append(line_fitx)
    line.bestx = np.mean(line.recent_xfitted,axis=0)
    line.best_fit = np.polyfit(liney, line.bestx, 2)
    line.current_fit = line_fit
    line.radius_of_curvature = get_radius_curvature(best_fit)
    line.line_base_pos = abs(img.shape[1] / 2 - linex_base) * xm_per_pix 
    line.allx = linex
    line.ally = liney

    return line

def get_radius_curvature(line_fit):
    ploty = np.linspace(0, 719, num=720)

    #bottom of image, ie where the car is 
    y_eval = np.max(ploty)

    plotx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    fit_world = np.polyfit(ploty*ym_per_pix, plotx*xm_per_pix, 2)
    
    curverad = ((1 + (2*fit_world[0]*y_eval*ym_per_pix + fit_world[1])**2)**1.5) / np.absolute(2*fit_world[0])
    return curverad