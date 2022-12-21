# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:19:56 2022

@author: kbons
"""


import numpy as np
import gdal
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt    
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import joblib 

# define input raster and output raster path
inpRaster = 'C:/Users/kbons/Desktop/GAMA Land Cover Change/new project/Ground Truth/GAMA_2022_cmpst.tif' 
outRaster = 'C:/Users/kbons/Desktop/GAMA Land Cover Change/new project/Ground Truth/Classified/RF_GAMA_2022_fn.tif'

# Read Raster Data 
ds = gdal.Open(inpRaster)

# Retrieve raster attributes
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount
gt = ds.GetGeoTransform()
proj = ds.GetProjection()

# Read Raster as Array
array = ds.ReadAsArray() #(bands, rows, cols)

#modify structure by stacking bands to dorm one element
array = np.stack(array,axis=2) #(rows, cols, bands)
array = np.reshape(array, [rows*cols,bands]) # reshape to a 2d array so that it can match with the training data
array_df = pd.DataFrame(array, dtype='int16') # convert array to dataframe to keep both test and training data in the same structure


# Read  training data
gdf = gpd.read_file("C:/Users/kbons/Desktop/GAMA Land Cover Change/new project/Ground Truth/2022_truth.shp")
class_names = gdf['Label'].unique() # get class names
print ("class names", class_names)
class_ids = np.arange(class_names.size)+1 # assign ids to class names
print('class ids', class_ids)

df = pd.DataFrame({'Label': class_names, 'id': class_ids}) #create a dataframe of the class names and class ids
#df.to_csv("GAMA_2020 data/class_lookup.csv") # save dataframe as csv for future reference
print('gdf without ids', gdf.head())
gdf['class_id'] = gdf['Label'].map(dict(zip(class_names, class_ids))) #add class ids to the shapefile
print('gdf with ids', gdf.head())

# divide truth data data into test and train data
gdf_train = gdf.sample(frac=0.7)
gdf_test = gdf.drop(gdf_train.index)
print('gdf shape', gdf.shape, 'training shape', gdf_train.shape, 'test', gdf_test.shape)
gdf_train.to_file("C:/Users/kbons/Desktop/GAMA Land Cover Change/new project/Ground Truth/2022_train.shp")
gdf_test.to_file("C:/Users/kbons/Desktop/GAMA Land Cover Change/new project/Ground Truth/2022_test.shp")



#enter features to used for training according how they are named in the columns of training data
data = gdf_train[['b1_GAMA_22', 'b2_GAMA_22', 'b3_GAMA_22', 'b4_GAMA_22', 'b5_GAMA_22',
           'b6_GAMA_22', 'b7_GAMA_22']]
#enter training label according to your csv column name
label = gdf_train['class_id']


data_test = gdf_test[['b1_GAMA_22', 'b2_GAMA_22', 'b3_GAMA_22', 'b4_GAMA_22', 'b5_GAMA_22',
           'b6_GAMA_22', 'b7_GAMA_22']]

label_test = gdf_test['class_id']



####no need to modify the code below###
#######################################


#set classifier parameters and train classifier
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(data,label)

#predict classes
y_pred = clf.predict(array_df)


classification = y_pred.reshape((rows,cols)) #reshape predicted classes into a 2d array

# Display map
def color_image_show(img, title):
    fig = plt.figure(figsize=(15,15))
    fig.set_facecolor('white')
    plt.imshow(img)
    plt.title(title)
    plt.show()
    

# display image
color_image_show(classification, 'GAMA Random Forest 1991')

# write classified image as a tiff file
def createGeotiff(outRaster, data, geo_transform, projection):
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Int32)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    rasterDS = None



#export classified image
createGeotiff(outRaster,classification,gt,proj)

#Accuracy assessment
clf.score(data_test,label_test) #check performance of classifier on test data
clf.score(data,label)  #check performance of classifier on train data
# classification report
x_pred= clf.predict(data_test)
print(classification_report(label_test, x_pred, target_names=class_names))

#confusion matrix
cm = confusion_matrix(label_test, x_pred)
pd.DataFrame(cm, index=class_names, columns=class_names)

kappa= cohen_kappa_score(label_test, x_pred)
kappa

#joblib.dump(clf, "RF_L8_11")

#mj = joblib.load("C:/Users/kbons/Desktop/Classification Scripts/RF_L8_11")
