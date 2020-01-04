import numpy as np
from matplotlib import pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tqdm import tqdm
import read_d
import pandas as pd

test_path = '../test'
csv_filepath = 'submission.csv'
res_path = 'submission.csv'
new_test = read_d.new_test(csv_filepath,test_path)
model = load_model('model.h5')
y_pred = model.predict(new_test)[:,1]

csv_file = csv.reader(open(csv_filepath))
name = []
for cont in csv_file:
    if cont[0]!='Id':
        name.append(cont[0])
dataframe = pd.DataFrame({'Id':name,'Predicted':y_pred})
dataframe.to_csv(res_path,index=False)
