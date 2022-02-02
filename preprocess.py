import urllib.request
import zipfile
import pandas as pd

urllib.request.urlretrieve("https://timeseriesclassification.com/Downloads/ECG200.zip",'ecg200.zip')

with zipfile.ZipFile('ecg200.zip', 'r') as zip_ref:
    zip_ref.extractall('ecg200/')

train = pd.read_csv('ecg200/ECG200_TRAIN.txt', header=None, sep='\n')
train = train[0].str.split('  ',expand=True).iloc[:,1:]
train = train.applymap(float)
print('train_data : ')
train.head(10)

test = pd.read_csv('ecg200/ECG200_TEST.txt', header=None, sep='\n')
test = test[0].str.split(r"\s+",expand=True).iloc[:,1:]
test = test.applymap(float)
print('test_data : ')
test.head(10)

train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)
print('csv saved')
