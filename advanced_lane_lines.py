import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


cv2.namedWindow( "Overall", cv2.WINDOW_AUTOSIZE );
test_images = os.listdir("test_images/")


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

def binarize_image(img, sobel_kernel = 3, sx_thresh =(20,100),s_thresh=(170,255) , sdir_thresh = (0,np.pi/2)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    gray = hls[:,:,2]

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

    s_binary = np.zeros_like(sdir_binary) 
    s_binary[(gray >= s_thresh[0]) & (gray <= s_thresh[1])] = 1

    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) 
    #color_binary = np.dstack(( sdir_binary, sxbinary, s_binary)) 
    overall_binary = np.zeros_like(sdir_binary)
    #overall_binary[(s_binary==1) & (sdir_binary==1) & (sxbinary==1)] = 1
    overall_binary[(s_binary==1) | (sxbinary==1)] = 1


    #cv2.imshow("color_binary", color_binary)
    #cv2.imshow("Overall", overall_binary)
    #cv2.imshow("sxbinary",sxbinary)
    #cv2.imshow("s_binary", s_binary)
    #cv2.imshow("sdir_binary", sdir_binary)
    #plt.imshow(s_binary)
    #plt.waitforbuttonpress()
    #cv2.waitKey(0)
    return overall_binary



if __name__ == "__main__":
    camera_mtx, distortion = calibrate_camera(9,6,"camera_cal/")

    #img = mpimg.imread("camera_cal/calibration1.jpg")
    #cv2.imshow("raw",img)
    #undist = cv2.undistort(img,camera_mtx,distortion,camera_mtx, None)
    #cv2.imshow("undist",undist)
    #cv2.imwrite("writeup_files/calibration.jpg",undist)
    #cv2.waitKey(0)
    

    for i,x in enumerate(test_images):
        img = cv2.imread("test_images/" + x)
        #cv2.imshow("Raw",img)
        undist = cv2.undistort(img,camera_mtx,distortion,camera_mtx, None)
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
        binary = binarize_image(undist,sobel_kernel = 3, sx_thresh =(20,100),s_thresh=(170,255),sdir_thresh = (0.95,1.0))
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
        
        # polygon [(224,634), (489,484), (1026,634), (802,484)]
        # corrected []
        print(warped.shape)
        cv2.imshow("top down",warped)
        cv2.waitKey(0)
        









    






