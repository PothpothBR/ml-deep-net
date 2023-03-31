import tensorflow as tf
def min_max(data, column):
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())     
    return data

def color(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label