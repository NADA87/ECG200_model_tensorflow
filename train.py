#FROM tensorflow/tensorflow:2.7.0


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
import tensorflow_addons as tfa
from joblib import load, dump

import os
 
# Get the list of all files and directories
path = "./"
os.listdir(path)
print(os.listdir(path))

df = pd.read_csv('./train.csv')

print('dataframe shape :',df.shape)
df.head(3)

print(' 1 : Normal')
print('-1 : Myocardial Infarction')
df['1'].value_counts()

plt.plot(df.iloc[1,1:])

plt.plot(df.iloc[0,1:])

a = []
for i in range(0,df.shape[0]): a = a + list(df.iloc[i,1:])

print('max electric signal recorded :', max(a))
print('min electric signal recorded :', min(a))
print('mean electric signal recorded :', np.mean(a))
print('std electric signal recorded :', np.std(a))

df1 = df.iloc[:,1:]

tf.keras.backend.clear_session()

input1 = tf.keras.layers.Input(shape=(96,1))  

conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu',strides=1)(input1)
drop1 = tf.keras.layers.Dropout(0.5)(conv1)
pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
flat1 = tf.keras.layers.Flatten()(pool1)

conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu',strides=1)(input1)
drop2 = tf.keras.layers.Dropout(0.5)(conv2)
pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop2)
flat2 = tf.keras.layers.Flatten()(pool2)

# conv3 = tf.keras.layers.Conv1D(filters=7, kernel_size=12, activation='relu', strides=1)(input1)
# drop3 = tf.keras.layers.Dropout(0.5)(conv3)
# pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop3)
# flat3 = tf.keras.layers.Flatten()(pool3)

# conv4 = tf.keras.layers.Conv1D(filters=7, kernel_size=24, activation='relu',strides=1)(input1)
# drop4 = tf.keras.layers.Dropout(0.5)(conv4)
# pool4 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop4)
# flat4 = tf.keras.layers.Flatten()(pool4)

concate = tf.keras.layers.Concatenate()([flat1,flat2])

batch_norm = tf.keras.layers.BatchNormalization()(concate)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(batch_norm)

model = tf.keras.Model(inputs=input1, outputs=outputs)

model.summary()

model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), optimizer='adam', metrics=[tf.keras.metrics.Precision(thresholds=0.5),'accuracy'])

callback = tf.keras.callbacks.ModelCheckpoint(
    './saved_model/', monitor='val_accuracy', verbose=0, save_best_only=True,
    save_freq='epoch'
)

Model = model.fit(df1,np.array([1.0 if i==1 else 0.0 for i in df['1'] ]) , 
                  epochs=100, batch_size=16, validation_split= 0.25 ,
                  callbacks = [callback])


#loded_model = tf.keras.models.load_model('./saved_model/')

test = pd.read_csv('./test.csv')
truth_ = test['1']
test = test.iloc[:,1:]


#train_cr = classification_report( [1.0 if i==1 else 0.0 for i in df['1'] ]  ,[ float(i[0]>0.5) for i in  loded_model.predict(df1) ])
#test_cr = classification_report( [1.0 if i==1 else 0.0 for i in truth_ ]  ,[ float(i[0]>0.5) for i in  model.predict(test) ] )




#dump({'model_history':Model.history,
#      'test_classification_report':test_cr},'model_result.joblib')

model_result = {
    'model_history' : Model.history}

dump(model_result,'./model_result.joblib')

report = classification_report([1.0 if i==1 else 0.0 for i in truth_ ], [ float(i[0]>0.5) for i in  model.predict(test) ], output_dict=True)
report = pd.DataFrame(report).transpose()
report['index'] = report.index
dump(report,'./report.joblib')
















