## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

## Camera Calibration
* The code for this part of the project is in camera_calibrate.py
* For camera calibration you need two arrays, one to hold where the points should be in 3D space (`objpoints`) and one for where the `cv2.findChessboardCorners` detects 2D points in the image (`imgpoints`). The objpoints have x,y,z coordinates and are generated using the numpy `mgrid` function. For each image if `cv2.findChessboardCorners` correctly finds the corners in the image new object points and image points are appended to `objpoints` and `imgpoints` array; but first the image points are corrected using the `cornerSubPix` function to reduce the re-projection error. With the two arrays generated I call `cv2.calibrateCamera` to get the camera matrix and distortion coefficients. The reprojection error I get is 0.83 which is within the acceptable range. 
* With the camera matrix, distortion coefficients and `cv2.undistort` function  you can undistort images. Below shows an example image undistored using the method 

![Alt text](https://github.com/scheideman/CarND-Advanced-Lane-Lines/blob/master/writeup_files/calibration1.jpg?raw=true "Calibration Image")


## Pipeline
1. First undistort image using steps above
  * TODO: Image goes here
2. Get a binary image, see `binarize_image` function in advanced_lane_lines.py
 * To get a binary image I used a combination of thresholding on the sobel filter applied in both the x, y directions, and      thresholding on the saturation and lighness channels of the HLS colorspace.  
 * TODO: Example image
3. Use perspective transform to get top-down view of lane lines
 * TODO explanation and example image
4. Find lane lines by fitting polynomial to identified pixels
 * TODO explanation and image
5. Find radius of curvature and position of the car with respect 
---
#### Video


---

## Discussion
