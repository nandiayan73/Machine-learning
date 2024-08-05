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
# cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = train_set, validation_data = test_set, epochs = 25)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/predictions/test4.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
# training_set.class_indices
print(result)
if result[0][0] == 0:
    prediction = 'Jack'
else:
    prediction = 'Oggy'
print(prediction)




