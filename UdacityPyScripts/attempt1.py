
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
from skimage.color import rgb2gray
from skimage import transform

def load_data(data_directory):
    directories=[]
    for i in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, i)):
            directories.append(i)

    labels=[]
    images=[]
    for i in directories:
        label_directory=os.path.join(data_directory, i)
        file_names=[]
        for j in os.listdir(label_directory):
            if j.endswith(".ppm"):
                file_names.append(os.path.join(label_directory,j))

        for k in file_names:
            images.append(skimage.data.imread(k))
            labels.append(int(i))
    return images, labels





image, label=load_data("C:\\Users\\yashm\\Downloads\\Training")
#image=np.array(image)
#label=np.array(label)
"""


unique_labels=set(label)
plt.figure(figsize=(15,15))
x=1
for i in unique_labels:
    img=image[label.index(i)]
    plt.subplot(8,8,x)
    plt.axis('off')
    plt.title("Label {0}, ({1})".format(i+1, label.count(i)))
    x=x+1
    plt.imshow(img)
plt.show()
"""

imagedit = [transform.resize(i, (50, 50)) for i in image]
imagedit=np.array(imagedit)
imagedit=rgb2gray(imagedit)

traffic_signs=[124,633,879,963]
for i in range(len(traffic_signs)):
    plt.subplot(1,4,i+1)
    plt.axis("off")
    plt.imshow(imagedit[traffic_signs[i]], cmap="gray")
plt.show()
