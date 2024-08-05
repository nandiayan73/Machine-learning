import numpy as np
from keras.preprocessing import image

test_image = image.load_img('test1.avif', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(test_image)