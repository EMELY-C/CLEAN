# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import rdkit.Chem as Chem
from rdkit.Chem import rdMolHash
import rdkit.Chem.AllChem as AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random 
import pubchempy as pc
from sklearn.model_selection import cross_val_score, KFold

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
import joblib

robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
robjects.r('''source('rcdk.R')''')
get_fingerprint = robjects.globalenv['get_fingerprint']
get_descriptors = robjects.globalenv['get_descriptors']
# from pycdk.pycdk import MolFromSmiles, getFingerprint
def get_cdk_fingerprints(smi):
    types=['standard', 'pubchem', 'kr', 'maccs', 'estate', 'circular']
    fps = []
    for tp in types:
        fps += list(get_fingerprint(smi, tp))
    return fps
def get_fp_(smi,tp):
    return list(list(get_fingerprint(smi, tp)))
def get_descriptors_(smi):
    return list(get_descriptors(smi))
# different maching learning algorithm for predicting fingerprints
def svm_linear(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)   
    return accuracy, precision,recall, f1


def svm_linear_2024(X, y, model_filename="svm_linear_model.pkl"):
    # 分割数据集成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # 创建SVM分类器
    classifier = SVC(kernel='linear', random_state=0)

    # 训练模型
    classifier.fit(X_train, y_train)

    # 预测测试集
    y_pred = classifier.predict(X_test)

    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 保存模型
    joblib.dump(classifier, model_filename)
    # dump(classifier, model_filename)  # 如果你使用的是 joblib

    return accuracy, precision, recall, f1

def svm_linear2(X,y,X1,y1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X1)
    accuracy=accuracy_score(y1,y_pred)
    precision=precision_score(y1,y_pred)
    recall=recall_score(y1,y_pred)
    f1=f1_score(y1,y_pred)
    return accuracy, precision,recall, f1
def svm_poly(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = SVC(kernel = 'poly', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1
def svm_rbf(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1
def logi(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1
def bay(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1
def dec(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1
def ran(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state = 0 )
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1

def knn5(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier =KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)  
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1
def ann(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = Sequential()
    classifier.add(Dense(units =40, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    return accuracy, precision,recall, f1
# train the data using different maching learning algorithm and return accuracy, precision,recall and F1 score for all used fingerprints
def mach_lea(spe,fp,algorithm):
    acc_=[]
    pre_=[]
    rec_=[]
    f1_=[]
    for i in range(fp.shape[1]):
        acc1,pre1,rec1,f11=algorithm(spe,fp[:,i])
        acc_.append(acc1)
        pre_.append(pre1)
        rec_.append(rec1)
        f1_.append(f11)
    df=(pd.DataFrame([acc_,pre_,rec_,f1_])).T
    df.columns=("acc","pre","rec","f1")
    return df
# test dataset
def mach_lea2(spe,fp,spe1,fp1,algorithm):
    acc_=[]
    pre_=[]
    rec_=[]
    f1_=[]
    for i in range(fp.shape[1]):
        acc1,pre1,rec1,f11=algorithm(spe,fp[:,i],spe1,fp1[:,i])
        acc_.append(acc1)
        pre_.append(pre1)
        rec_.append(rec1)
        f1_.append(f11)
    df=(pd.DataFrame([acc_,pre_,rec_,f1_])).T
    df.columns=("acc","pre","rec","f1")
    return df
# shuffle each digit of fingerprint of all compounds and train the data
def shuffle(spe,fp,algorithm):
    acc_=[]
    pre_=[]
    rec_=[]
    f1_=[]
    for i in range(fp.shape[1]):
        list_=fp[:,i].copy()
        random.shuffle(list_)
        acc1,pre1,rec1,f11=algorithm(spe,list_)
        acc_.append(acc1)
        pre_.append(pre1)
        rec_.append(rec1)
        f1_.append(f11)
    df=(pd.DataFrame([acc_,pre_,rec_,f1_])).T
    df.columns=("acc","pre","rec","f1")
    return df
# predict fingerprints using hss vector using svm algorithm
def svm_(X,y,hss):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier.predict(hss)
# Get score based on predicted fingerprints and real fingerprints 
def score(fp_pre,fp_,df):
    count_f1=0
    for i in range(fp_pre.shape[1]):
        if(fp_pre[0,i]==fp_[0,i]):
            count_f1=count_f1+df["f1"][i]
    return count_f1

# exclude stereoisomer and standardize the SMILES using RDkit  
def iso(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=0)
# search pubchem using formula and return a list of matched SMILES
def searchpubchem(formula):
    try:    
        smileslist= pc.get_properties("CanonicalSMILES",formula,"formula",list_return='flat')
        smiles=[]
        for i in smileslist:
            smiles.append(i["CanonicalSMILES"])
        return smiles
    except:
        return []

def SmilestoFormula(smi):
    try:
        mol=Chem.MolFromSmiles(smi)
        return rdMolHash.MolHash(mol,rdMolHash.HashFunction.MolFormula)
    except:
        return None
def is_formula(formula):
    formula=formula.replace("Cl","A")
    ele=["C","H","A","F","N","O","P","S","0","1","2","3","4","5","6","7","8","9"]
    for i in formula:
        if(i not in ele):
            return 0
    return 1
def read_inhouse_database(filename):
    in_house_database=pd.read_csv(filename)
    list_smi_database=list((in_house_database["SMILES"]))
    for i in range(len(list_smi_database)):
        try:
            list_smi_database[i]=iso(list_smi_database[i])
        except:
            list_smi_database[i]=None
    list_smi_database=list(set(list_smi_database))
    list_formula=[]
    i=0
    while(i<len(list_smi_database)):        
        formula_i=SmilestoFormula(list_smi_database[i])
        if(formula_i==None or "." in list_smi_database[i]):
            del list_smi_database[i] 
            continue
        if(is_formula(formula_i)):
            list_formula.append(formula_i)
            i=i+1
        else:
            del list_smi_database[i]
    df1=pd.DataFrame({"SMILES":list_smi_database,"Formula":list_formula})
    dic1={}
    for i in range(len(df1)):
        dic1.setdefault(df1["Formula"][i],[]).append(df1["SMILES"][i]) 
    return dic1
def search_inhouse(formula,df_database):
    df=df_database[df_database["Formula"]==formula]
    return list(df["SMILES"])
def SmilestoMW(smiles):
    s=Chem.MolFromSmiles(smiles)
    d=Descriptors.ExactMolWt(s)
    countp=smiles.count("+]")+2*smiles.count("+2]")+3*smiles.count("+3]")+4*smiles.count("+4]")
    countn=smiles.count("-]")+2*smiles.count("-2]")+3*smiles.count("-3]")+4*smiles.count("-4]")
    return d-1.00728*(countp-countn+1)      
def compare_structure(smiles1, smiles2):
    if(Chem.MolFromSmiles(smiles1)==None or Chem.MolFromSmiles(smiles2)==None):
        return 0
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)    
    fp1 = getfp(smiles1)
    fp2 = getfp(smiles2)
    return DataStructs.DiceSimilarity(fp1, fp2)

# save model
import joblib
def mach_lea_1(spe,fp):
    for i in range(fp.shape[1]):
        svm_linear_1(spe,fp[:,i],i)#output model
# save model
def svm_linear_1(X,y,i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    filename="svm"+str(i)+".model"
    f=open(filename,"wb")
    joblib.dump(classifier,f)
    f.close()

def knn_(X,y,hss):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    return classifier.predict(hss)


def knn_cv(X, y):
    # 创建 KNeighborsClassifier 实例
    classifier = KNeighborsClassifier(n_neighbors=5)

    # 进行 10 倍交叉验证
    cv = KFold(n_splits=10, shuffle=True, random_state=0)

    # 用于存储每一折的评分
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # 计算每一折的评分
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')  # 使用加权平均
        recall = recall_score(y_test, y_pred, average='weighted')  # 使用加权平均
        f1 = f1_score(y_test, y_pred, average='weighted')  # 使用加权平均

        # 存储每一折的评分
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # 打印每一折的结果
        print(
            f'Fold results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # 输出交叉验证的平均值和标准差
    print(f'\nAverage Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}')
    print(f'Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}')
    print(f'Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}')
    print(f'Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')

# 确保在调用 knn5 函数之前，将 X 和 y 设置为特征和标签数据