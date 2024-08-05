import tensorflow as tf
rank1_tensor=tf.Variable(["Hare","Krishna"],tf.string)
rank2_tensor=tf.Variable([["Radhe","Radhe","Jai"],["Krishna","Krishna","jai"],["Haribol","Jai ho","Jai"]])

t=tf.zeros([5,5,5,5])

t=tf.reshape(t,[125,-1])