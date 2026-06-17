import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf

import cv2
fashion_mnist = tf.keras.datasets.mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

import matplotlib.pyplot as plt
img=train_images[7590]
img_resize=cv2.resize(img,(280,380),interpolation=cv2.INTER_CUBIC)
plt.imshow(img_resize,cmap='gray')
plt.axis('off')
plt.show()