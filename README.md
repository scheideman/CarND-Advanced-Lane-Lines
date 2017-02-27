## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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

![Alt text](https://github.com/scheideman/CarND-Advanced-Lane-Lines/blob/master/output_images/calibration1.jpg?raw=true "Calibration Image")


## Pipeline
1. First undistort image using steps above. Below is a undistorted sample image.
![Alt text](https://github.com/scheideman/CarND-Advanced-Lane-Lines/blob/master/output_images/straight_lines1.jpg?raw=true "Undistored image")
2. Get a binary image, see `binarize_image` function in advanced_lane_lines.py
 * To get a binary image I used a combination of thresholding on the sobel filter applied in both the x, y directions, and      thresholding on the saturation and lighness channels of the HLS colorspace.  
![Alt text](https://github.com/scheideman/CarND-Advanced-Lane-Lines/blob/master/output_images/binary_test1.jpg?raw=true "Binary image")
3. Use perspective transform to get top-down view of lane lines
 * Lines 18-22 in advanced_lane_lines.py shows the code for the perspective transform
 * A top-down view makes it easier to fit lines to the curve, to get a top down view you can use the function `cv2.getPerspectiveTransform`
 * For the perpective transform I hardcoded the following `src` and `dst` points
   * `src = np.float32([(600,450), (700,450), (200,720),(1150,720)`
    * `dst = np.float32([(300,0), (1000,0), (300,720), (1000,720)])`
![Alt text](https://github.com/scheideman/CarND-Advanced-Lane-Lines/blob/master/output_images/topdown_straight_lines2.jpg?raw=true "Topdown image")
 
4. Find lane lines by fitting polynomial to identified pixels
 * The `get_line` function in lane_tracking.py implements this functionality
 * To find the lane lines a sliding window search was used, starting at the two peaks of the histogram of the bottom half of   the image.
 * 9 windows were used with a width of `margin`, with their centers adjusted if greater than `min_pix` lane pixels fell within the window. Then all the pixels identified in those windows were used to fit a line with the `np.polyfit` function
 * After the lane lines are found in one frame, for the next frame the same line is used with all the pixels falling within a `margin` around the posterior line used to calculate the new line. This functionality is in the `update_line` function in lane_tracking.py
![Alt text](https://github.com/scheideman/CarND-Advanced-Lane-Lines/blob/master/output_images/lane_line2_4185.39704572.jpg?raw=true "sliding window image")
5. Find radius of curvature and position of the car with respect ?raw=true "Sliding window")
 * With the polynomial fit parameters for the line found in step four, the radius of curvature is calculated in the `get_radius_curvature` function in lane_tracking.py. The bottom of the image is the y value used for calculating the radius of curvature 
 * The cars offset from the center of the lane is also calculated on line 135 in advanced_lane_finding.py, using the x positions of the fit line at the bottom of the image.
 * Both the radius of curvature and center offset are display on the output image in meters as seen below 
![Alt text](https://github.com/scheideman/CarND-Advanced-Lane-Lines/blob/master/output_images/result_straight_lines2.jpg?raw=true "radcurve image")
* The `process_image` function in advanced_lane_line.py shows the complete pipeline implementation

---
#### Video
https://youtu.be/7ua9s3tet0c

---

## Discussion
