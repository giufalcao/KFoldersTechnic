# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:06:37 2019

@author: Falcao
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
from os import listdir
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def scores(file, clf_name, metodo, target_test, prediction, output):
        with open(output, 'at') as out_file:
            line = f"\"{file} - {clf_name} - {metodo} - Treino \n \","
            line += f"{accuracy_score(target_test, prediction)},"
            line += f"{matthews_corrcoef(target_test, prediction)},"
            line += f"{f1_score(target_test, prediction,average='macro')},"
            line += f"{recall_score(target_test, prediction, average='macro')},"
            line += f"{precision_score(target_test, prediction, average='macro')}\n"
            out_file.writelines(line)
            #print(f"{classification_report(self.treated_data.target_test, self.prediction)}")
        pass

dir = '/datasets/'
output = 'output.csv'
names=[]
acuracia = [] 
precision = []
mcc = []
fscore = []

for file in listdir(dir):
    # Le arquivo CSV, para pandas dataframe
    data = pd.read_csv(dir + file, comment='@', header=None)
    
    with open(output, 'wt') as out_file: 
        out_file.writelines('\"Descrição\",\"Acurácia\",\"F1-Score\",\"Recall\",\"Precisão,MCC\"\n')

    # transforma dados categóricos em dados numéricos ()
    encoder = LabelEncoder()
    data = data.apply(encoder.fit_transform)

    # transforma o dataframe em uma matriz com os features (ft) e 
    # cria um vetor com os alvos/targets (tg), nome das classes 
    ft = data.iloc[:, 0:-1]
    tg = data.iloc[:,-1]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    ft = scaler.fit_transform(ft)
    
    training = KFold(n_splits=10, random_state=42, shuffle=False)
    
    for train_index, test_index in training.split(ft):
        ft_train, ft_test, tg_train, tg_test = ft[train_index], ft[test_index], tg[train_index], tg[test_index]
    
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(ft_train, tg_train)
        pred = dt.predict(ft_test)
        #precision.append(dt.precision(ft_test, tg_test))   
        scores(file, "Decision Tree", "MinMaxScaler", tg_test, pred, output)
     
        naive_bayes = GaussianNB()
        naive_bayes.fit(ft_train, tg_train)
        pred = naive_bayes.predict(ft_test)
        scores(file, "Naive Bayes", "MinMaxScaler", tg_test, pred, output)
        
        rfc = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        rfc.fit(ft_train, tg_train)
        pred = rfc.predict(ft_test)
        scores(file, "Random Forest", "MinMaxScaler", tg_test, pred, output)
        
        svclassifier = SVC(kernel='linear')  
        svclassifier.fit(ft_train, tg_train)
        pred = svclassifier.predict(ft_test)  
        scores(file, "SVC", "MinMaxScaler", tg_test, pred, output)
