from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint,Callback
from densenet import get_model
import tensorflow as tf
import numpy as np
import read_d
import keras

train_path = "../train_val" 
label_path = "../train_val.csv"
final_output_dir = "model.h5"
X_train_new,x_val, train_label, y_val=read_d.new_data(train_path,label_path)
model = get_model(weights=None)
model.compile(loss='categorical_crossentropy', optimizer='adamax',metrics=["accuracy"])

reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2,patience=3, min_lr=0.001)
checkpoint = ModelCheckpoint(final_output_dir,monitor='val_loss',verbose=0,save_best_only=True,mode='min',period=1)
callbacks_list=[reduce_lr,
                checkpoint,
               ]
model.fit(X_train_new,train_label,batch_size=30,epochs=20,validation_data=(x_val,y_val),callbacks=callbacks_list,shuffle=True)
