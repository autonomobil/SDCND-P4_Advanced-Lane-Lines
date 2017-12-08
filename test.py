import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from matplotlib import animation, rc
import matplotlib.animation as animation

from IPython.display import HTML, Image
import glob
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

image_count = 0
images = glob.glob('./video_images/*.jpg')
images = sort_nicely(images)

img = mpimg.imread(images[0])

fig = plt.figure(figsize=(11,6))

ax1 = plt.subplot(2,2,1)
im1 = plt.imshow(img, animated=True)

ax2 = plt.subplot(2,2,2)
im2 = plt.imshow(img, animated=True)

ax3 = plt.subplot(2,2,3)
im3 = plt.imshow(img, animated=True)
plt.title('left lane')

ax4 = plt.subplot(2,2,4)
im4 = plt.imshow(img, animated=True)
plt.title('right lane')
plt.axis('off')

def updatefig(*args):
    global image_count
    image_count += 1
    img = mpimg.imread(images[image_count])

    im1.set_array(img)
    im2.set_array(img)
    im3.set_array(img)
    im4.set_array(img)
    return im1, im2, im3, im4,

ani = animation.FuncAnimation(fig, updatefig, interval=5, blit=True)
ani
plt.show()
