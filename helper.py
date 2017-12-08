#edge enhacement
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

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

def norm_0_255(img):
    img_min = np.min(img)
    img_max = np.max(img)
    range_img = img_max - img_min
    result = (img-img_min)/range_img*255

    return(np.asarray(result, dtype='uint8'))

images = images = glob.glob('./video_images/*.jpg')

with open('calibration_file.p', mode='rb') as handle:
    cali_file = pickle.load(handle)
mtx = cali_file['mtx']
dist = cali_file['dist']

def undistort_img(img, mtx = mtx, dist = dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def warp_img(img, src = None, dst = None, w_Mat = [0]):
    ysize = img.shape[0]
    xsize = img.shape[1]
    w_Mat_inv = None
    print(img.shape)

    if len(w_Mat) == 1:
        w_Mat = cv2.getPerspectiveTransform(src,dst)
        w_Mat_inv = cv2.getPerspectiveTransform(dst,src)

    img = cv2.warpPerspective(img, w_Mat,(xsize,ysize), flags=cv2.INTER_LINEAR)

    return img, w_Mat, w_Mat_inv

def edge_enhancement(img, max_value=8):

    kernel = np.array([[-1,-1,-1,-1,-1],
                         [-1,2,2,2,-1],
                         [-1,2,max_value,2,-1],
                         [-1,2,2,2,-1],
                         [-1,-1,-1,-1,-1]]) / max_value

    output = cv2.filter2D(img,-1,kernel)

    return output

def sharpen(img):
    kernel = np.array([[-1,-1,-1],
                         [-1,9,-1],
                         [-1,-1,-1]])

    # kernel = np.array([[1,1,1],
    #                      [1,-7,1],
    #                      [1,1,1]])
    output = cv2.filter2D(img,-1,kernel)

    return output

ysize = 720
xsize = 1280
#source points
src = np.float32([(681,445),    #top right
                (602,445),      #top left
                (258,690),      #bottom left
                (1063,690)])    #bottom right

# destination points
dst = np.float32([(xsize-425,0),
                (425,0),
                (425,ysize),
                (xsize-425,ysize)])

images = sort_nicely(images)



for i,img_name in enumerate(images[1020:1025]):

    img = mpimg.imread(img_name)
    img = undistort_img(img)

    warped_e = edge_enhancement(img,8)
    warped_e,_,_= warp_img(warped_e, src =src, dst=dst)

    img, w_Mat, w_Mat_inv = warp_img(img, src =src, dst=dst)

    warped_s = sharpen(img)

    # warped_e = edge_enhancement(warped_s,8)
    warped_s = sharpen(img)
    warped_es = sharpen(edge_enhancement(img))

    plt.figure()
    plt.subplots_adjust(top=0.98,bottom=0.0,left=0.0,right=1.0,hspace=0.1,wspace=0.05)
    plt.subplot(2,2,1)
    plt.imshow(norm_0_255(img))
    plt.title('original')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(norm_0_255(warped_e))
    plt.title('warped_e')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(norm_0_255(warped_s))
    # plt.imshow(norm_0_255(output))
    # plt.imshow(output)
    plt.title('warped_s')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(norm_0_255(warped_es))
    plt.title('warped_es')
    plt.axis('off')

    plt.show()
