# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:11:52 2022

@author: nurul
"""
#%% import section 

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay


from tensorflow.keras import Sequential,Input
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization


#%%path section

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.data'
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

#modelpath
mms_path=os.path.join(os.getcwd(),'model','mms.pkl')
OHE_PATH=os.path.join(os.getcwd(),'model','ohe.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model','model.h5')

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#%% step 1) data loading

# load dataset

df = pd.read_csv(url,delim_whitespace=(True) , na_values='?',header=None)

#delim_whitespace is used to separate the columns in the dataset

#%%step 2) data inspection

df.info()
df.head()
df.isna().sum() # check up the nan/missing value
#Glucose,BloodPressure,SkinThickness,BMI have nan
df.describe().T # to get basic statistics of the data

#rename the columns
df.columns=['surgery?','Age','Hospital_Number','rectal_temperature','pulse','respitory_rate',
            'extremities','peripheral_pulse','membranes','refill_time','pain',
            'peristalsis',' distension','tube','reflux','reflux_PH','examination_feces','abdomen',
            'cell_volume',' total_protein', 'abdominocentesis_appearance','abdominocentesis_total_protein','outcome',
            'surgical_lesion?','type_of_lesion1','type_of_lesion2','type_of_lesion3','cp_data']
print(df.head())
df.isna().sum() # check up the nan/missing value
#All vars imvolve of Nan except Age,hospital_number,surgical_lesion?,type_of_lesion1,type_of_lesion2,type_of_lesion3,cp_data

#Con Var
con_col=['rectal_temperature','pulse','respitory_rate','reflux_PH',
         'cell_volume',' total_protein','abdominocentesis_total_protein']
#Cat Var
cat_col=df.drop(labels=con_col, axis=1).columns

#cat_col
for i in cat_col:
    plt.figure()
    sns.countplot(df[i])
    plt.show()
#con_col    
for i in con_col:
    plt.figure()
    sns.distplot(df[i])
    plt.show()


#checking up outlier
df.boxplot()

#%%step 3) data cleaning

#drop the unnecessary col
df=df.drop(labels=['Hospital_Number','type_of_lesion2','type_of_lesion3','reflux_PH'],axis=1)

from sklearn.impute import KNNImputer

#Con_var/using knn imputer
columns_name=df.columns

knn_i=KNNImputer()
df=knn_i.fit_transform(df)
df=pd.DataFrame(df)
df.columns=columns_name
df.isna().sum()


cat_col=list(cat_col&df.columns) #to select  both of it

for i in cat_col:
    df[i]=np.floor(df[i]).astype(int)

df.info()

df.duplicated().sum()
#cat will be in float/object
#con will be in int

#%%
# step 4) features selection
#con vs cat
X=df.drop(labels='outcome',axis=1)
y=df['outcome'].astype(int)

con_col=['rectal_temperature','pulse','respitory_rate',
         'cell_volume',' total_protein','abdominocentesis_total_protein']

selected_features=[]

for i in con_col:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),y) # X(continous), Y(Categorical)
    print(i)
    print(lr.score(np.expand_dims(df[i],axis=-1),y))
    if lr.score(np.expand_dims(X[i],axis=-1),y)>0.6:
        selected_features.append(i)
        
print(selected_features)

# cat vs cat 
#cramers v

for i in cat_col:
    print(i)
    matrix=pd.crosstab(df[i],y).to_numpy()
    print(cramers_corrected_stat(matrix))
    if cramers_corrected_stat(matrix)>0.4:
        selected_features.append(i)
        
print(selected_features)

#For the conclusion, only cell_volume and type of lesion 1 were been picked up for the model development

df= df.loc[:,
 selected_features]
X=df.drop(labels='outcome',axis=1)
y=df['outcome'].astype(int)


#%% step 5) data preprocessing

#mms scaler

mms=MinMaxScaler()
X=mms.fit_transform(X)

#deeplearning using OHE for our target

ohe=OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=-1))

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                   random_state=123)


#%% model development

#for classification issue/can used the np.unique

nb_class=len(np.unique(y,axis=0))

#step 1) create a container
model=Sequential()
model.add(Input(shape=np.shape(X_train)[1:]))
model.add(Dense(128,activation='relu',name='1st_hidden_layer'))
#nodes is 5 we decided
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu',name='2nd_hidden_layer'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(nb_class,activation='softmax'))#output_layer
model.summary()

#%%


tensorboard_callback=TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
#early_callback=EarlyStopping(monitor='val_acc',patience=5)


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

hist=model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),callbacks = [tensorboard_callback])

#%% Model Evaluation
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('epoch')
plt.legend(['Training Acc','Validation Acc'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['Training loss','Validation loss'])
plt.show()

#confusionmatrix
pred_y=model.predict(X_test)
pred_y=np.argmax(pred_y,axis=1)
true_y=np.argmax(y_test,axis=1)

cm=confusion_matrix(true_y,pred_y)
cr=classification_report(true_y, pred_y)

#displaytheconfusionmatrix
labels=["lived","dead","was euthanized"]
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
#printthematrix
print(cr)

#%%model saving

#mms saving
with open(mms_path,'wb') as file:
    pickle.dump(mms,file)

#ohe saving
with open(OHE_PATH,'wb') as file:
    pickle.dump(OHE_PATH,file)
    
#save model.h5
model.save(MODEL_SAVE_PATH)