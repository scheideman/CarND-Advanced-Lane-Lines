import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip

test_images = os.listdir("test_images/")
undistorted_images = os.listdir("undistorted_images/")

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def calibrate_camera(nx,ny,file_path):
    camera_cal_images = os.listdir(file_path)

    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # set x,y coordinats. z is always zero

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i,x in enumerate(camera_cal_images):
        img = mpimg.imread(file_path + x)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,(nx,ny), None)

        if ret:
            objpoints.append(objp)

            #increase accuracy
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            #cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
            #cv2.imshow("Corners",img)
            #cv2.waitKey(250)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None) 
    
    print("Camera calibration status: {}".format(ret))

    return mtx, dist

def binarize_image(img, sobel_kernel = 3, sx_thresh =(20,100),sy_thresh =(25,255),s_thresh=(170,255) , sdir_thresh = (0,np.pi/2), hue_thresh = (20,50),lightness_thresh = (150,255)):
    
    #cv2.imshow("raw",img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    saturation = hls[:,:,2]
    hue = hls[:,:,0]
    lightness = hls[:,:,1]
    
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,sobel_kernel) 
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,sobel_kernel)

    abs_sobelx = np.absolute(sobelx)  
    abs_sobely = np.absolute(sobely)  

    dir_gradient = np.arctan2(abs_sobely,abs_sobelx) 
    sdir_binary = np.zeros_like(dir_gradient)
    sdir_binary[(dir_gradient >= sdir_thresh[0]) & (dir_gradient <= sdir_thresh[1])] = 1
    #convert to 8 bit
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(sdir_binary) 
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1 
    
    scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    sybinary = np.zeros_like(sdir_binary) 
    sybinary[(scaled_sobel >= sy_thresh[0]) & (scaled_sobel <= sy_thresh[1])] = 1 

    #kernel = np.ones((5,5),np.uint8)
    #sybinary = cv2.dilate(sybinary,kernel,iterations = 1)


    s_binary = np.zeros_like(sdir_binary) 
    s_binary[(saturation >= s_thresh[0]) & (saturation <= s_thresh[1])] = 1
    #kernel = np.ones((5,5),np.uint8)
    #s_binary = cv2.dilate(s_binary,kernel,iterations = 1)


    hue_binary = np.zeros_like(sdir_binary)
    hue_binary[ (hue >= hue_thresh[0]) & (hue <= hue_thresh[1])] = 1

    lightness_binary = np.zeros_like(sdir_binary)
    lightness_binary[ (lightness >= lightness_thresh[0]) & (lightness <= lightness_thresh[1])] = 1

    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) 
    #color_binary = np.dstack(( sdir_binary, sxbinary, s_binary)) 
    overall_binary = np.zeros_like(sdir_binary)
    #overall_binary[(s_binary==1) & (sdir_binary==1) & (sxbinary==1)] = 1
    overall_binary[((sybinary==1) & (sxbinary==1)) | (((s_binary==1) & (lightness_binary==1))==1)] = 1

    #cv2.imshow("color_binary", color_binary)
    #cv2.imshow("Overall", overall_binary)
    #cv2.imshow("sxbinary",sxbinary)
    #cv2.imshow("s_binary", s_binary)
    #cv2.imshow("hue_binary", hue_binary)
    #cv2.imshow("light_binary", lightness_binary)
    #cv2.imshow("sy_binary", sybinary)
    #cv2.imshow("sdir_binary", sdir_binary)
    #plt.imshow(s_binary)
    #plt.waitforbuttonpress()
    #cv2.waitKey(0)
    return overall_binary



def find_lane_lines(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    out_img = np.dstack((img, img, img))*255

    midpoint = np.int(histogram.shape[0]/2)

    # TODO check if max is greater than some threshold to be a line
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.waitforbuttonpress()
    #plt.close()
    return left_fit, right_fit

def update_line_from_next_frame(img,left_fit, right_fit):
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

def get_radius_curvature(line_fit):
    ploty = np.linspace(0, 719, num=720)

    #bottom of image, ie where the car is 
    y_eval = np.max(ploty)

    plotx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    fit_world = np.polyfit(ploty*ym_per_pix, plotx*xm_per_pix, 2)
    
    curverad = ((1 + (2*fit_world[0]*y_eval*ym_per_pix + fit_world[1])**2)**1.5) / np.absolute(2*fit_world[0])
    return curverad
    


camera_mtx, distortion = calibrate_camera(9,6,"camera_cal/")
src = np.float32([(600,450), (700,450), (200,720),(1150,720)])
dst = np.float32([(300,0), (1000,0), (300,720), (1000,720)])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst,src)
TRACKING = False

def process_image(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    undist = cv2.undistort(image,camera_mtx,distortion,camera_mtx, None)
    binary = binarize_image(undist,sobel_kernel = 3, sx_thresh =(10,255),lightness_thresh = (100,255), sy_thresh=(25,255),s_thresh=(130,255))
    warped = cv2.warpPerspective(binary, M, (1280,720)) 
    left_fit, right_fit = find_lane_lines(warped)
    if(TRACKING is False):
        left_fit, right_fit = find_lane_lines(warped)
    
    # won't work binary image
    return warped


if __name__ == "__main__":
    #camera_mtx, distortion = calibrate_camera(9,6,"camera_cal/")

    #img = mpimg.imread("camera_cal/calibration1.jpg")
    #cv2.imshow("raw",img)
    #undist = cv2.undistort(img,camera_mtx,distortion,camera_mtx, None)
    #cv2.imshow("undist",undist)
    #cv2.imwrite("writeup_files/calibration.jpg",undist)
    #cv2.waitKey(0)

    white_output = 'project_video_test.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    
    for i,x in enumerate(test_images):
        #img = cv2.imread("test_images/" + x)
        #cv2.imshow("Raw",img)
        #undist = cv2.undistort(img,camera_mtx,distortion,camera_mtx, None)
        #cv2.imwrite("undistorted_images/" + x,undist)
        print(x)
        undist = cv2.imread("undistorted_images/"+ x)

        #cv2.imshow("Undist",undist)

        hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)
        # dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3)) 
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        #cv2.imshow("H",hsv[:,:,0])
        #cv2.imshow("S",hsv[:,:,1])
        #cv2.imshow("V",hsv[:,:,2])
        #gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("GRAY",S)
        binary = binarize_image(undist,sobel_kernel = 3, sx_thresh =(10,255),lightness_thresh = (100,255), sy_thresh=(25,255),s_thresh=(130,255))
        #plt.imshow(binary)
        #plt.waitforbuttonpress()
        #TL, TR, BL, BR
        src = np.float32([(600,450), (700,450), (200,720),(1150,720)])
        #dst = np.float32([(224,484), (800+224,484), (224,150+484), (800+224,150+484)])
        dst = np.float32([(300,0), (1000,0), (300,720), (1000,720)])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst,src)
        #warped = cv2.warpPerspective(binary, M, (binary.shape[1],binary.shape[0])) 
        warped = cv2.warpPerspective(binary, M, (1280,720)) 

        left_fit, right_fit = find_lane_lines(warped)
        print("left: {0}".format(get_radius_curvature(left_fit)))
        print("right: {0}".format(get_radius_curvature(right_fit)))
