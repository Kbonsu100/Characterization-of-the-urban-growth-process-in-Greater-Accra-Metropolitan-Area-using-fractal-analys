# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 10:44:57 2023

@author: kbons
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 12:21:17 2022

@author: kbons
"""

#import required packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import gdal

#read images
img1 = cv2.imread("High_fd/1991.tif", 0)
img2 = cv2.imread("High_fd/2002.tif", 0)
img3 = cv2.imread("High_fd/2013.tif", 0)
img4 = cv2.imread("High_fd/2022.tif", 0)


# fig, axs = plt.subplots(1,4, figsize=(10,10))
# axs[0].imshow(img1)
# axs[1].imshow(img2)
# axs[2].imshow(img3)
# axs[3].imshow(img4)
# plt.show()


#display images
fig, ax = plt.subplots(figsize= (10,10))
ax.imshow(img1)
ax.axis("off")
plt.show()


#mathematical morphology


#structuring element
kernel = np.ones((10,10), np.uint8)

#Erosion
erosion = cv2.erode(img1, kernel, iterations =1)

#Dilation
dilation = cv2.dilate(img1, kernel, iterations =1)

#Opening
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)

#Closing
closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)

#Gradient
gradient = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)

#1991 Morphological Analysis
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
kernel = np.ones((17,17), np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((12,12), np.uint8)
erosion_91 = cv2.erode(closing, kernel, iterations =2)

fig, ax = plt.subplots(figsize= (10,10))
ax.imshow(erosion_91)
ax.axis("off")
plt.show()

#2002 Morphological Analysis
kernel = np.ones((10, 10), np.uint8)
opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
kernel = np.ones((40,40), np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((15,15), np.uint8)
erosion_02 = cv2.erode(closing, kernel, iterations =1)

fig, ax = plt.subplots(figsize= (10,10))
ax.imshow(erosion_02)
ax.axis("off")
plt.show()

#2013 Morphological Analysis
kernel = np.ones((10, 10), np.uint8)
opening = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
kernel = np.ones((50,50), np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((27,27), np.uint8)
erosion_13 = cv2.erode(closing, kernel, iterations =1)

fig, ax = plt.subplots(figsize= (10,10))
ax.imshow(erosion_13)
ax.axis("off")
plt.show()

#2022 Morphological Analysis
kernel = np.ones((10, 10), np.uint8)
opening = cv2.morphologyEx(img4, cv2.MORPH_OPEN, kernel)
kernel = np.ones((50,50), np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((33,33), np.uint8)
erosion_22 = cv2.erode(closing, kernel, iterations =1)

fig, ax = plt.subplots(figsize= (10,10))
ax.imshow(erosion_22)
ax.axis("off")
plt.show()

#save
img = gdal.Open("High_fd/2022.tif")
driverTiff = gdal.GetDriverByName("Gtiff")
rows = img.RasterYSize
cols = img.RasterXSize
gt = img.GetGeoTransform()
proj = img.GetProjection()
n = img.RasterCount



driver = gdal.GetDriverByName("GTiff")
driver.Register()
output = driver.Create("UCS_22.tif", cols, rows, 1, gdal.GDT_Float32)
output.SetGeoTransform(gt) 
output.SetProjection(proj) 
output.GetRasterBand(1).SetNoDataValue(0)
output.GetRasterBand(1).WriteArray(erosion_22)
output.FlushCache()
output = None
