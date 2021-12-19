# KMeans and SOM on an image to identify dominant colors (clusters) to represnt the image
# Author: Boris Kundu
# ## Import Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from minisom import MiniSom

#Install minisom
#pip install minisom

#Read image
img = plt.imread('dominant-colors.jpg')

#Display Image
plt.title('Original Image')
plt.imshow(img)

# # Image Pre-processing
# ## Reshape Data
#Check original shape
print(f'Original Image Shape:{img.shape}')
#Reshape to get data
data = img.reshape(img.shape[0]*img.shape[1],img.shape[2])
#Check shape of dataset
print(f'Image Dataset Shape:{data.shape}')
print(f'Sample Data:{data[0,:]}')

# ## Standardize Data
scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
print(f'Sample Scaled Data:{scaled_data[0,:]}')

# # Identifying best K for K-Means (Elbow Method & Silhouette Score)
#Sum of Squared Errors
sse = []
#Silhouette Score
slc = []
for k in range(2,7):
    #print(f'\nK:{k}')
    kmeans = KMeans(n_clusters=k,init='random')
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)
    #print(f'\nSum of Sum of Squared Error:{sse[-1]}')
    #slc.append(silhouette_score(scaled_data, kmeans.labels_))
    #print(f'\nSilhouette Score:{slc[-1]}')

#Plot error vs K
K = [i for i in range(2,7)]
fig,axes = plt.subplots(figsize=(10,5),num='Elbow Method')
axes.set_title("Elbow Method")
axes.set_ylabel("Sum of Squared Errors")
axes.set_xlabel("K")
axes.set_xticks(K)
axes.plot(K,sse,marker='o')
plt.show()

#Plot silhouette score vs K
#fig,axes = plt.subplots(figsize=(10,5),num='Silhouette Method')
#axes.set_title("Silhouette Method")
#axes.set_ylabel("Silhouette Score")
#axes.set_xlabel("K")
#axes.set_xticks(K)
#axes.plot(K,slc,marker='x')
#plt.show()

print(f'Sum of Squared Errors:{sse}')
#print(f'Silhouette Scores:{sse}')

# # KMeans Implementation
# Number of clusters
kmeans = KMeans(n_clusters=4,init='random')
# Fitting the input data
kmeans = kmeans.fit(scaled_data)
# Get cluster labels
kmeans_clusters = kmeans.predict(scaled_data)
# Get centroid values
kmeans_centroids = kmeans.cluster_centers_

# ## KMeans Dominant Colors
#Check kmeans parameters
print(f'KMeans paramerters:\n{kmeans}')
#Check kmeans iterations for convergence
print(f'KMeans iterations:{kmeans.n_iter_}')
#Check sum of squared errors
print(f'KMeans sum of squared errors:{kmeans.inertia_}')
#Check kmeans centroids
print(f'KMeans Dominant Colors(Centroids):\n{kmeans_centroids}')

#Create KMeans dominant color image
kmeans_domcolimg = []
for i in range(4):
    for j in range(400):
        kmeans_domcolimg.append(kmeans_centroids[i])
kmeans_domcolimg = np.array(kmeans_domcolimg)
#print(kmeans_domcolimg.shape)
#Inverse transformation
domcol_kmeans = scaler.inverse_transform(kmeans_domcolimg)
#Reshape
domcolimg_kmeans = domcol_kmeans.reshape(40,40,3)
#Display KMeans dominant colors
fig,axes = plt.subplots(figsize=(10,5),num='KMeans Dominant Colors')
axes.set_title("KMeans Dominant Colors")
axes.imshow(domcolimg_kmeans.astype(int))
plt.show()

#Recreate image using KMeans clusters and dominant colors (centroids)
kmeans_image = []
for cluster in kmeans_clusters:
    kmeans_image.append(kmeans_centroids[cluster])
kmeans_image = np.array(kmeans_image)

#Inverse transformation
data_kmeans = scaler.inverse_transform(kmeans_image)
#Reshape
img_kmeans = data_kmeans.reshape(img.shape[0],img.shape[1],img.shape[2])
print(f'KMeans - Image shape after inverse transformation and reshape:{img_kmeans.shape}')

#Display Image with KMeans dominant colors
fig,axes = plt.subplots(figsize=(10,5),num='KMeans Dominant Colors')
axes.set_title("Image with KMeans Dominant Colors")
axes.imshow(img_kmeans.astype(int))
plt.show()

# # Identifying best perceptron plane for SOM

#Get best plane for perceptron(s)
error = []
plane_x = 1 
plane_y = 1
least_error = 1
for i in range(1,10):
    for j in range (1,10):
        som = MiniSom(i, j, scaled_data.shape[1], sigma=0.1, learning_rate=0.1, random_seed=101)
        som.train(scaled_data, 10000)
        qunatinzation_error = som.quantization_error(scaled_data)
        print(f'Perceptron plane:({i},{j}) has quantization error:{qunatinzation_error}')
        error.append(qunatinzation_error)
        if qunatinzation_error < least_error:
            least_error = qunatinzation_error
            plane_x = i
            plane_y = j

print(f'Perceptron plane:({plane_x},{plane_y}) has least quantization error:{least_error}')

#Plot Error vs K
planes = [i+1 for i in range(len(error))]
fig,axes = plt.subplots(figsize=(10,10),num='SOM')
axes.set_title("SOM")
axes.set_ylabel("Quantization Error")
axes.set_xlabel("Perceptron Plane")
axes.plot(planes,error,marker='o')
plt.show()

# # SOM Implementation
som = MiniSom(3, 8, scaled_data.shape[1], sigma=0.1, learning_rate=0.1,random_seed=101)
som.train(scaled_data, 10000, verbose=True)

# ## SOM Dominant Colors
#Perform quantization
scaled_data_qnt=som.quantization(scaled_data)
print(f'SOM Dominant Colors (Perceptron Weights):\n{som._weights}')
#Inverse transform
data_qnt=scaler.inverse_transform(scaled_data_qnt)
#Reshape to get image
som_image=data_qnt.reshape(img.shape[0],img.shape[1],3)

#Create SOM dominant color image
som_domcolimg = []
centroids = som._weights
for cent in centroids:
    for c in cent:
        for j in range(384):
            som_domcolimg.append(c)
som_domcolimg = np.array(som_domcolimg)
print(som_domcolimg.shape)
#Inverse transformation
domcol_som = scaler.inverse_transform(som_domcolimg)
#Reshape
domcolimg_som = domcol_som.reshape(96,96,3)
#Display KMeans dominant colors
fig,axes = plt.subplots(figsize=(10,5),num='SOM Dominant Colors')
axes.set_title("SOM Dominant Colors")
axes.imshow(domcolimg_som.astype(int))
plt.show()

#Display Image with SOM dominant colors
fig,axes = plt.subplots(figsize=(10,5),num='SOM Dominant Colors')
axes.set_title("Image with SOM Dominant Colors")
axes.imshow(som_image.astype(int))
plt.show()