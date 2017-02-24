import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

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

if __name__ == "__main__":

    camera_mtx, distortion = calibrate_camera(9,6,"camera_cal/")

    img = mpimg.imread("camera_cal/calibration1.jpg")
    cv2.imshow("raw",img)
    undist = cv2.undistort(img,camera_mtx,distortion,camera_mtx, None)
    cv2.imshow("undist",undist)
    cv2.imwrite("writeup_files/calibration.jpg",undist)
    cv2.waitKey(0)
