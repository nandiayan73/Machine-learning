
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator



# preprocessing of data
print(dir)

train_datagen=ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

train_set=train_datagen.flow_from_directory('dataset/traininig_set',
                                                 target_size = (64, 64),
                                                 batch_size = 2,
                                                 class_mode = 'binary')                                   
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 2,
                                            class_mode = 'binary')                                  

# Building the cnn:                                    

cnn=tf.keras.Sequential()

# 1st convolution layer:s
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2nd convolution layer:s
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full connection:
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer:
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = train_set, validation_data = test_set, epochs = 25)



import cv2
# Load an image from file
image = cv2.imread('download.jpg')

cv2.imshow('Face Detection', image)



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


face_images = []

# Loop through the detected faces and extract each one
for (x, y, w, h) in faces:
    face_roi = image[y:y + h, x:x + w]
    # print(face_roi)
    face_images.append(face_roi)


print(face_images)

import numpy as np
from keras.preprocessing import image

# training_set.class_indices

# Display or save the individual face images
for i, face in enumerate(face_images):
    cv2.imwrite(f'face_{i+1}.jpg', face)
    test_image = image.load_img(f'face_{i+1}.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    if(result[0][0]==1):
        cv2.imshow('Jack', face)
    else:
        cv2.imshow('Oggy', face)

cv2.waitKey(0)
cv2.destroyAllWindows()