from net import Net
from scallers import min_max, color

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd

batch_size = 128

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train = ds_train.map(color, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)

ds_test = ds_test.map(color, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()

df_train: pd.DataFrame = tfds.as_dataframe(ds_train, ds_info)
df_test: pd.DataFrame = tfds.as_dataframe(ds_test, ds_info)

print(df_train.shape)
df_train_batchs = [df_train.iloc[i:i+batch_size] for i in range(0, df_train.shape[0]-batch_size, batch_size)]
df_train_batchs.append(df_train.iloc[len(df_train_batchs)*batch_size:])

df_train.reset_index()

for i in df_train.iterrows():
    print(i[2])

exit()
n = Net([8, 6, 4, 1], 0.08)

for entry, expect in zip(data.values, results.values):
    n.fit(entry, expect)
    n.save("diabetes")
