import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from camera_calibrate import calibrate_camera
from moviepy.editor import VideoFileClip
from lane_tracking import find_lane_lines, update_line_with_new_frame, Line

test_images = os.listdir("test_images/")
undistorted_images = os.listdir("undistorted_images/")

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

camera_mtx, distortion = calibrate_camera(9,6,"camera_cal/")
src = np.float32([(600,450), (700,450), (200,720),(1150,720)])
dst = np.float32([(300,0), (1000,0), (300,720), (1000,720)])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst,src)
TRACKING = False

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



def get_radius_curvature(line_fit):
    ploty = np.linspace(0, 719, num=720)

    #bottom of image, ie where the car is 
    y_eval = np.max(ploty)

    plotx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    fit_world = np.polyfit(ploty*ym_per_pix, plotx*xm_per_pix, 2)
    
    curverad = ((1 + (2*fit_world[0]*y_eval*ym_per_pix + fit_world[1])**2)**1.5) / np.absolute(2*fit_world[0])
    return curverad
    
def visualize_lane(left_fit, right_fit, img, warped):

    ploty = np.linspace(0, 719, num=720)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result





def process_image(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    undist = cv2.undistort(image,camera_mtx,distortion,camera_mtx, None)
    binary = binarize_image(undist,sobel_kernel = 3, sx_thresh =(10,255),lightness_thresh = (100,255), sy_thresh=(25,255),s_thresh=(130,255))
    warped = cv2.warpPerspective(binary, M, (1280,720)) 
    left_fit, right_fit = find_lane_lines(warped)
    if(TRACKING is False):
        left_fit, right_fit = find_lane_lines(warped)


    left_radius = get_radius_curvature(left_fit)
    right_radius = get_radius_curvature(right_fit)

    result = visualize_lane(left_fit,right_fit, undist,warped)
    print("Left: ", left_radius)
    print("Right: ", right_radius)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    # won't work binary image
    return result


if __name__ == "__main__":


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
