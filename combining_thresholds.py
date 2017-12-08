import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import pickle
import glob
import re
from scipy.signal import argrelextrema
# from helper import *
images = glob.glob('./test_images/*.jpg')

ksize = 15
xsize_factor_top = 0.12
xsize_factor_bottom = 0
ysize_factor_top = 1.6
ysize_factor_bottom = 0.95

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    key_no = [tryint(c) for c in re.split('([0-9]+)', s) ]
    key_no = int(key_no[1])
    return key_no

def sort_nicely(liste):
    """ Sort the given list in the way that humans expect.
    """
    liste.sort(key=alphanum_key)
    return liste

images = sort_nicely(images)

############ functions
def norm_0_255(img):
    if img.max() <= 1 or img.max() > 255 or img.min() <= 0  :
        img_min = np.min(img)
        img_max = np.max(img)
        range_img = img_max - img_min
        result = (img-img_min)/range_img*255
        return(np.asarray(result, dtype='uint8'))

    else:
        print("+++WARNING Normalizing to range 0 - 255: image already in range 0-255")
        return img

def norm_0_1(img):
    if img.max() > 1 or img.max() < 1 or img.min() <= 0:
        img_min = np.min(img)
        img_max = np.max(img)
        range_img = img_max - img_min
        result = (img-img_min)/range_img
        return(np.asarray(result, dtype='float32'))
    else:
        print("+++WARNING Normalizing to range 0 - 1: image has already a maximum <= 1")
        return img

def warp_img(img, src = None, dst = None, w_Mat = [0]):
    ysize = img.shape[0]
    xsize = img.shape[1]
    w_Mat_inv = None

    if len(w_Mat) == 1:
        w_Mat = cv2.getPerspectiveTransform(src,dst)
        w_Mat_inv = cv2.getPerspectiveTransform(dst,src)

    img = cv2.warpPerspective(img,w_Mat,(xsize,ysize), flags=cv2.INTER_LINEAR)

    return img, w_Mat, w_Mat_inv

def weighted_img(initial_img, alpha, img, beta, λ=0.):

    if initial_img.ndim == 2:
        temp = np.zeros_like(initial_img)
        initial_img = np.dstack((initial_img, temp, temp))
    if img.ndim == 2:
        temp = np.zeros_like(img)
        img = np.dstack((img, temp, temp))

    return cv2.addWeighted(initial_img, alpha, img, beta, λ)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:

        if (img.max() <= 1 or img.max() > 255) and img.min() <= 0  :
            ignore_mask_color = 1
        elif (img.max()) > 1:
            ignore_mask_color = 255

    vertices  = np.array(vertices, ndmin = 3, dtype = np.int32)
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    # print("max masked image", masked_image.max())
    # print("min masked image", masked_image.min())
    return masked_image

def colormask(img,c_mask_low,c_mask_high, return_cmask = 0, colorspace = None):
    """
    Apply a colormask on image and then return it
    """
    if colorspace == 'hsv_img':
        img_colorspaced= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    elif colorspace == 'hls':
        img_colorspaced= cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    elif colorspace == 'lab':
        img_colorspaced = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    else:
        img_colorspaced = img
    # Threshold the HSV image to get only yellow/white
    c_mask = cv2.inRange(img_colorspaced, c_mask_low, c_mask_high)

    if return_cmask:
        return np.asarray(c_mask, dtype='uint8')

    img_c_masked = cv2.bitwise_and(img, img, mask=c_mask)

    return img_c_masked

def colormask_treshold(img, lower, upper):
    """
    Apply a colormask on image and then return it
    """

    mask =(img[:,:,0] >= lower[0]) & (img[:,:,0] <= upper[0]) \
        & (img[:,:,1] >= lower[1]) & (img[:,:,1] <= upper[1]) \
        & (img[:,:,2] >= lower[2]) & (img[:,:,2] <= upper[2])

    return mask

def win_adap_tresh(img, winsize_y =200, winsize_x=400):

    # limits = np.concatenate((lower, upper))
    # print(limits)
    # factor_mean = limits[adap_what]
    ysize = img.shape[0]
    xsize = img.shape[1]
    mean_img = np.mean(img)
    mask = np.zeros_like(img)

    # print(" ")
    # print("mean_img:", mean_img)
    for y_win in range(0, ysize, winsize_y):
        win_y_low = y_win
        win_y_high = y_win + winsize_y
        if win_y_high > ysize:
            win_y_high = ysize

        # print(' ')
        # print("y win", win_y_low, win_y_high)

        for x_win in range(0, xsize, winsize_x):
            win_x_low = x_win
            win_x_high = x_win + winsize_x
            if win_x_high > xsize:
                win_x_high = xsize

            # print("x win", win_x_low, win_x_high)

            # mean_win = np.mean(orig_img[win_y_low:win_y_high, win_x_low:win_x_high, :])
            # print("mean_win:", mean_win)
            # limits[adap_what] = min(int(mean_win * factor_mean),253)
            # print("adap_what",adap_what)
            # print("limits[adap_what]:", limits[adap_what])
            # print(' ')

            mask[win_y_low:win_y_high, win_x_low:win_x_high] = cv2.adaptiveThreshold(img[win_y_low:win_y_high, win_x_low:win_x_high]
                                                                ,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,-7)

            # mask[win_y_low:win_y_high, win_x_low:win_x_high] = \
            #     (img[win_y_low:win_y_high, win_x_low:win_x_high, 0] >= limits[0]) \
            #     & (img[win_y_low:win_y_high, win_x_low:win_x_high, 0] <= limits[3]) \
            #     & (img[win_y_low:win_y_high, win_x_low:win_x_high, 1] >= limits[1]) \
            #     & (img[win_y_low:win_y_high, win_x_low:win_x_high, 1] <= limits[4]) \
            #     & (img[win_y_low:win_y_high, win_x_low:win_x_high, 2] >= limits[2]) \
            #     & (img[win_y_low:win_y_high, win_x_low:win_x_high, 2] <= limits[5])
            # cv2.rectangle(mask,(384,0),(510,128),(0,255,0),3)

    return mask

def sobel_thresh(images, orient='x', ksize=11, thresh=(30, 255)):

    if len(images.shape) <= 2:
        images = np.array([images])

    binary = np.zeros_like(images[0])
    binary[0,0] =0

    for i, img in enumerate(images):
        if orient == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        elif orient == 'y':
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        # scaled_sobel = norm_0_255(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        binary_temp = np.zeros_like(scaled_sobel)
        binary_temp[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        binary = (binary==1) | (binary_temp==1)

    return binary

def adaptiveThreshold(images, block_size= 111, mean_value = 1):

    if len(images.shape) <= 2:
        images = np.array([images])

    binary_output = np.zeros_like(images[0])
    binary_output[0,0] =0

    for i, img in enumerate(images):

        # binary_output_temp = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,-mean_value)
        binary_output_temp = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,mean_value)

        binary_output = (binary_output==1) | (binary_output_temp==1)

    # Return the binary image
    return binary_output

def get_binary_img(img, plot = 0):
    orig_img = img.copy()
    ysize = img.shape[0]
    xsize = img.shape[1]

    # flags = img[img > 0]

    mean_img = np.mean(img)
    print("mean_img: ", mean_img)
    #####
    # img = gaussian_blur(img, 3)

    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls1_img = hls_img[:,:,1]
    hls2_img = hls_img[:,:,2]

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hls1_img = hsv_img[:,:,1]
    hsv2_img = hsv_img[:,:,2]

    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab0_img = lab_img[:,:,0]
    lab2_img = lab_img[:,:,2]

    red_img = img[:,:,0]

    #####

    gradx = np.asarray(sobel_thresh(np.array([red_img, hsv2_img, hls2_img]), orient='x', ksize = -1, thresh=(25, 255)), dtype='float32')
    gradx[645:720,0:280] = 0
    gradx[645:720,1000:1280] = 0
    gradx = gaussian_blur(gradx, 13)
    gradx[gradx >= 0.003] = 1
    gradx[gradx < 1] = 0
    gradx = np.asarray(gradx, dtype='uint8')

    adap_mask = np.asarray(win_adap_tresh(hls2_img, winsize_y = 90, winsize_x=160), dtype='uint8')
    # adap_mask = np.asarray(adaptiveThreshold(np.array([ lab2_img]), block_size= 61, mean_value = -5), dtype='uint8')
    # adap_mask = np.asarray(adaptiveThreshold(np.array([red_img, hsv2_img, lab2_img]), block_size= 61, mean_value = -5), dtype='uint8') #| np.asarray(adaptiveThreshold(np.array([red_img, hsv2_img])), dtype='uint8')
    adap_mask[645:720,0:280] = 0
    adap_mask[645:720,1000:1280] = 0
    #####
    # Threshold image to get only yellow/white
    ### H L S
    lower_yel = np.array((15,0,115))
    upper_yel = np.array((25,255,255))
    lower_wht = np.array((0, min(int(mean_img * 2.5),210),  10))
    upper_wht = np.array((255, 255, 255))

    hls_yel = colormask_treshold(hls_img, lower_yel,upper_yel)
    hls_wht = colormask_treshold(hls_img, lower_wht,upper_wht)

    hls_mask = np.asarray(hls_wht | hls_yel, dtype='uint8')

    ### L A B
    lower_yel = np.array((5,127,155))
    upper_yel = np.array((255,135,255))
    lower_wht = np.array((min(int(mean_img * 2.6),220), 125, 120))
    upper_wht = np.array((255, 140, 145))

    lab_yel = colormask_treshold(lab_img, lower_yel,upper_yel)
    lab_wht = colormask_treshold(lab_img, lower_wht,upper_wht)

    lab_mask =  np.asarray(lab_wht | lab_yel, dtype='uint8')

    ### H S V
    lower_yel = np.array((15,110,0))
    upper_yel = np.array((25,255,255))
    lower_wht = np.array((0, 0, min(int(mean_img * 2.4),220)))
    upper_wht = np.array((255,30,255))

    hsv_yel = colormask_treshold(hsv_img, lower_yel,upper_yel)
    hsv_wht = colormask_treshold(hsv_img, lower_wht,upper_wht)

    hsv_mask = np.asarray(hsv_wht | hsv_yel, dtype='uint8')

    ### RED

    lower_red = np.array((min(int(mean_img * 2.3),235), 40, 40))
    upper_red = np.array((255,255,255))
    red_mask = colormask_treshold(img, lower_red, upper_red)
    red_mask = np.asarray(red_mask, dtype='uint8')

    ####### Combine
    # mask1 = gradx | adap_mask
    combined = np.asarray(red_mask + hsv_mask + lab_mask + hls_mask + adap_mask + gradx, dtype='uint8')
    combined[combined<3] = 0
    combined[combined>=3] = 1

    # combined, w_Mat, w_Mat_inv = warp_img(combined, src, dst)

    ########## CROPPIMG

    # orig_img_cropped = weighted_img(region_of_interest(orig_img, np.array(src, dtype = np.int32)), 0.7, orig_img,0.3)
    orig_img_cropped = orig_img
    histogram = np.sum(combined[int(ysize/2):,:], axis=0)

    ############ PLOTTING
    if plot:
        if plot==2:
            plt.figure()
            plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=0.1,wspace=0.05)

            plt.subplot(2,3,1)
            plt.imshow(hls_img[:,:,0], cmap='inferno')
            plt.title('hls_img[:,:,0]')
            plt.axis('off')

            plt.subplot(2,3,2)
            plt.imshow(hls_img[:,:,1], cmap='inferno')
            plt.title('hls_img[:,:,1]')
            plt.axis('off')

            plt.subplot(2,3,3)
            plt.imshow(hls_img[:,:,2], cmap='inferno')
            plt.title('hls_img[:,:,2]')
            plt.axis('off')

            plt.subplot(2,3,4)
            plt.imshow(hls_wht, cmap='inferno')
            plt.title('hls_wht')
            plt.axis('off')

            plt.subplot(2,3,5)
            plt.imshow(hls_yel, cmap='inferno')
            plt.title('hls_yel')
            plt.axis('off')

            plt.subplot(2,3,6)
            plt.imshow(hls_mask, cmap='inferno')
            plt.title('hls_mask')
            plt.axis('off')

            plt.figure()
            plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=0.1,wspace=0.05)
            plt.subplot(2,3,1)
            plt.imshow(lab_img[:,:,0], cmap='inferno')
            plt.title('lab_img[:,:,0]')
            plt.axis('off')

            plt.subplot(2,3,2)
            plt.imshow(lab_img[:,:,1], cmap='inferno')
            plt.title('lab_img[:,:,1]')
            plt.axis('off')

            plt.subplot(2,3,3)
            plt.imshow(lab_img[:,:,2], cmap='inferno')
            plt.title('lab_img[:,:,2]')
            plt.axis('off')

            plt.subplot(2,3,4)
            plt.imshow(lab_wht, cmap='inferno')
            plt.title('lab_wht')
            plt.axis('off')

            plt.subplot(2,3,5)
            plt.imshow(lab_yel, cmap='inferno')
            plt.title('lab_yel')
            plt.axis('off')

            plt.subplot(2,3,6)
            plt.imshow(lab_mask, cmap='inferno')
            plt.title('lab_mask')
            plt.axis('off')

            plt.figure()
            plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=0.1,wspace=0.05)
            plt.subplot(2,3,1)
            plt.imshow(hsv_img[:,:,0], cmap='inferno')
            plt.title('hsv_img[:,:,0]')
            plt.axis('off')

            plt.subplot(2,3,2)
            plt.imshow(hsv_img[:,:,1], cmap='inferno')
            plt.title('hsv_img[:,:,1]')
            plt.axis('off')

            plt.subplot(2,3,3)
            plt.imshow(hsv_img[:,:,2], cmap='inferno')
            plt.title('hsv_img[:,:,2]')
            plt.axis('off')

            plt.subplot(2,3,4)
            plt.imshow(hsv_wht, cmap='inferno')
            plt.title('hsv_wht')
            plt.axis('off')

            plt.subplot(2,3,5)
            plt.imshow(hsv_yel, cmap='inferno')
            plt.title('hsv_yel')
            plt.axis('off')

            plt.subplot(2,3,6)
            plt.imshow(hsv_mask, cmap='inferno')
            plt.title('hsv_mask')
            plt.axis('off')

            plt.figure()
            plt.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0,hspace=0.1,wspace=0.05)
            plt.subplot(3,1,1)
            plt.imshow(red_img, cmap='inferno')
            plt.title('red_img')
            plt.axis('off')

            plt.subplot(3,1,2)
            plt.imshow(red_mask, cmap='inferno')
            plt.title('red_mask')
            plt.axis('off')

            plt.subplot(3,1,3)
            plt.imshow(adap_mask, cmap='inferno')
            plt.title('adap_mask')
            plt.axis('off')


            plt.show()
            #########

        plt.figure(facecolor='w', edgecolor='k')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        ax=[[],[],[],[],[],[],[],[]]
        ax[0] = plt.subplot(2,5,1)
        plt.title('original img')
        plt.axis('off')

        ax[1] = plt.subplot(2,5,2)
        plt.imshow(img, cmap='inferno')
        plt.title('img transformed')
        plt.axis('off')

        ax[2] =plt.subplot(2,5,3)
        plt.imshow(combined, cmap='inferno')
        plt.title('combined')
        plt.axis('off')
        #plt.plot(-histogram+ysize)

        ax[3] = plt.subplot(2,5,4)
        plt.title('result')
        plt.axis('off')

        plt.subplot(2,5,5)
        plt.imshow(gradx, cmap='inferno')
        plt.title('gradx')
        plt.axis('off')

        plt.subplot(2,5,6)
        plt.imshow(adap_mask, cmap='inferno')
        plt.title('adap_mask')
        plt.axis('off')

        plt.subplot(2,5,7)
        plt.imshow(red_mask, cmap='inferno')
        plt.title('red_mask')
        plt.axis('off')

        plt.subplot(2,5,8)
        plt.imshow(hls_mask, cmap='inferno')
        plt.title('hls_mask')
        plt.axis('off')

        plt.subplot(2,5,9)
        plt.imshow(lab_mask, cmap='inferno')
        plt.title('lab_mask')
        plt.axis('off')

        plt.subplot(2,5,10)
        plt.imshow(hsv_mask, cmap='inferno')
        plt.title('hsv_mask')
        plt.axis('off')

        # plt.show()

        plt.subplots_adjust(top=0.98,bottom=0.0,left=0.0,right=1.0,hspace=0.1,wspace=0.05)
    else:
        ax = 0

    return combined, ax
# [1030:]
for img_path in images[:]:
    img = mpimg.imread(img_path)
    orig_img = img.copy()
    print(img.shape)
    ysize = img.shape[0]
    xsize = img.shape[1]
    ploty = np.linspace(0, ysize-1, ysize)
    print(ploty)
    #source points
    src = np.float32([(682,445),
                    (601,445),
                    (247,690),
                    (1075,690)])

    # destination points
    dst = np.float32([(xsize-400,0),
                    (400,0),
                    (400,ysize),
                    (xsize-400,ysize)])

    img, w_Mat, w_Mat_inv = warp_img(img, src, dst)

    img, ax = get_binary_img(img, plot = 1)

    # Assuming you have created a warped binary image called "img"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(ysize/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(xsize/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # print("leftx_base", leftx_base)
    # print("rightx_base", rightx_base)
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(ysize/nwindows)
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
    minpix = 30
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # print(good_left_inds)
        # print(good_right_inds)
        # print('new window')
        # print('')
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

    out_img = norm_0_1(out_img)
    if not ax:
            plt.figure()
            ax = [[],[],[],[]]
            ax[0] = plt.subplot(221)
            ax[1] = plt.subplot(222)
            ax[2] = plt.subplot(223)
            ax[3] = plt.subplot(224)

    warped_img, _, _ = warp_img(orig_img, src, dst)

    ax[0].imshow(orig_img)
    ax[1].imshow(warped_img)
    ax[2].imshow(img, cmap='inferno')
    ax[3].imshow(out_img)
    plt.axis('off')
    plt.suptitle(img_path, fontsize=12)
    ax[3].plot(-histogram + ysize)
    ax[3].plot(left_fitx, ploty, color='yellow')
    ax[3].plot(right_fitx, ploty, color='yellow')

    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    plt.axis('off')
    plt.show()
