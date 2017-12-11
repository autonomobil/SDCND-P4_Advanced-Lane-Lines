[//]: # (Image References)

[img1]: ./output_images/1_undistorted1.png "1_undistorted1"
[img2]: ./output_images/1_undistorted2.png "1_undistorted1"
[img3]: ./output_images/2_img_straight.png "2_img_straight"
[img4]: ./output_images/2_img_straight2.png "2_img_straight2"
[img5]: ./output_images/3_bin_all_combined.png "all masks combined"
[img6]: ./output_images/3_bin_hls_mask.png "all masks combined"
[img7]: ./output_images/3_bin_hsv_mask.png "3_bin_hsv_mask"
[img8]: ./output_images/3_bin_lab_mask.png "3_bin_lab_mask"
[img9]: ./output_images/3_bin_red_and_adap_mask.png "3_bin_red_and_adap_mask"
[img10]: ./output_images/3_bin_result.png "3_bin_result"
[img11]: ./output_images/4_poly_img.png "4_poly_img"
[img12]: ./output_images/5_poly_search.png "5_poly_search"
[img13]: ./output_images/6_xm_ym_perpixel.png "6_xm_ym_perpixel"
[img14]: ./output_images/7_drawing_data.png "7_drawing_data"
[img15]: ./output_images/complete_pipeline.png "complete_pipeline"
[img16]: ./output_images/process_gif.gif "process_gif"

<!-- [image2]: ./output_images/example1.jpg "example from dataset"
[image3]: ./output_images/example2.jpg "example from dataset" -->

---
## SDCND Term 1 Project 4: Advanced Lane Lines
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project builds on the first project, but now uses advanced method of finding lines. The aim of this project is to write a robust lane line finding algorithm to deal with complex senarios like curving lines, shadows and changes in the color of the pavement.

**Results (Youtube link):**

[![result1](https://img.youtube.com/vi/6-ICQvQ6Y9A/0.jpg)](https://www.youtube.com/watch?v=6-ICQvQ6Y9A)

[![result2](https://img.youtube.com/vi/ELGEYoRz7Xo/0.jpg)](https://www.youtube.com/watch?v=ELGEYoRz7Xo)

The goals / steps of this project are the following:

1. **Calibration & Edge Enhancement**
    * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
    * Apply a distortion correction to raw images
    * Apply a edge enhancement to the images
2. **Warping & cutting image**
    * Warp with a perspective transform  ("birds-eye view")#
    * Cut out unnecessary parts of the image
3. **Getting Binary**
    * Getting the binary thresholded image by using color transforms, gradients, etc.
4. **Getting Lane Lines: Polyfit using sliding window**
    * Detect lane pixels and fit to find the lane boundary using the sliding window method
5. **Getting Lane Lines: Polyfit using previous fit search window**
    * Detect lane pixels and fit to find the lane boundary using the search window method
6. **Get meter per pixel**
    * Determine how much meter per pixel by using the distance from left to right lane line and dashed line length
7. **Measure Curvature and draw on real image**
    * Determine the curvature of the lane and vehicle position with respect to center
    * Warp the detected lane boundaries back onto the original image
8. **Complete Pipeline**

---

## Detailed Pipeline:

1. **Calibration & Edge Enhancement**

The OpenCV functions **"cv2.findChessboardCorners"** and **"cv2.calibrateCamera"** are used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository. The distortion matrix is than used to un-distort the images with the function **"cv2.undistort"**. Examples of undistorted calibration images:
![img1]
![img2]
A slight edge enhancement method is used, kernel comes from: *"OpenCV with Python By Example by Prateek Joshi"*

    kernel = np.array([[-1,-1,-1,-1,-1],
                         [-1,2,2,2,-1],
                         [-1,2,8,2,-1],
                         [-1,2,2,2,-1],
                         [-1,-1,-1,-1,-1]]) / 8

    output = cv2.filter2D(img,-1,kernel)

2. **Warping & cutting image**

The OpenCV function **"cv2.getPerspectiveTransform"** is been used to correctly rectify each image to a "birds-eye view". As the method of interpolation **"cv2.INTER_NEAREST"** gives the best result for my pipeline. The source and destination points are hand-tuned to the two images with straight lines, example:
![img3]

Then a "region of interest"-mask is used to cut out unnecessary areas (**"cv2.fillPoly"**), but keep enough of the image to see sharp curves, example:
![img4]
All of the above functions are summarized in a function called "preprocess_img", which undistorts, warps to bird-eye-view and than enhances the edges of the images

3. **Getting Binary**

This part is the most complex in the pipeline. The most important thing is: Different thresholding methods are used on the image and then a voting between these methods is done. Every method is split into recognizing yellow(*yel*) and white(*wht*) lines. This enables the algorithm (if used only on small windows of the image as described later) to detect the color of a line and once this is done, only look for the specified color. Used methods are:

* colormask threshold on HLS, HSV, LAB and red versions of the image; the tresholds are adaptive to the mean of the given image or window of image
* cv2.adaptiveThreshold is fed with multiple 1-channel images and then these maskes are connected with the Boolean AND
    * for white it is fed with the R-channel of the original image and the V-channel of the HSV image
    * for yellow it is fed with the S-channel of the HLS image and the B-channel of the LAB image

So in the end 5 different masks are generated, for the result they are added up and every pixel which is smaller 3 is set to 0 and every pixel which is bigger or equal 3 is set to 1 to get a normal binary image. The adaptive Threshold vote gets counted twice.

**H L S channels and mask:**
![img6]

**H S V channels and mask:**
![img7]

**L A B channels and mask:**
![img8]

**red channel and adaptive threshold mask:**
![img9]

**original image with boundaries for adaptive threshold, all masks combined, adaptive threshold mask:**
![img5]

![img16]


A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

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

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
