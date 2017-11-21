import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

image = mpimg.imread('signs_vehicles_xygrad.png')
ksize = 5

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    S = np.uint8(255*S/np.max(S))
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1

    return binary_output

image2 = hls_select(image, thresh=(45, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
f.tight_layout()
ax1.imshow(image, cmap = 'gray')
ax1.set_title('Original Image', fontsize=35)


def abs_sobel_thresh(img, orient='x', ksize=3, thresh=(0, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def mag_thresh(img, ksize=3, thresh=(0, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(sobelx*sobelx+sobely*sobely)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def dir_threshold(img, ksize=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grads = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    dir_grads_bin = np.zeros_like(dir_grads)
    dir_grads_bin[(dir_grads >= thresh[0]) & (dir_grads <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    binary_output = dir_grads_bin # Remove this line
    return binary_output


gradx = abs_sobel_thresh(image, orient='x', ksize=ksize, thresh=(15, 120))
grady = abs_sobel_thresh(image, orient='y', ksize=ksize, thresh=(15, 255))
mag_binary = mag_thresh(image, ksize=ksize, thresh=(60, 255))
dir_binary = dir_threshold(image, ksize=ksize, thresh=(-np.pi/2.4, np.pi/2.4))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1

print(image2.shape)
print(combined.shape)
image_final = cv2.addWeighted(image2, 0.5, combined, 0.5, 0)
ax2.imshow(image_final, cmap='gray')
ax2.set_title('combined', fontsize=35)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()
