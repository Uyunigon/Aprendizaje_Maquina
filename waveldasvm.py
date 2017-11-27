print '.....................Loading..........................................'
# ------------------------------------------------------------------------------
#                                   LIBRERIAS
# ------------------------------------------------------------------------------
# from numpy import *
import pywt
# import random
import scipy as sp
import numpy as np
import pylab as pl
from sklearn import svm
import matplotlib.pylab as pl
from scipy import linalg as la
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# # ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#                              DISCRETE WAVELET TRANSFORM
# ------------------------------------------------------------------------------

Dataset = np.loadtxt('AsignacionClases.csv', delimiter=',')
Dataset[0:,:21] = Dataset[:,:21]-Dataset[:,:21].mean(axis=0)
Dataset[0:,:21] = Dataset[:,:21]/Dataset[:,:21].var(axis=0)
databci = np.random.permutation(Dataset)
# databciw = np.random.permutation(Dataset)
# Discrete Wavelet Transform
# pywt.dwt(data, wavelet, mode='db2', axis=-1)
mode = 'db2'
cA, cD = pywt.dwt(databci, mode)
Wave = pywt.idwt(cA, cD, mode)
print'----------------------- VALORES DEL WAVELET -----------------------------'
print Wave
print'-------------------------------------------------------------------------'

# ------------------------------------------------------------------------------
#                                  FUNCION LDA
# ------------------------------------------------------------------------------
Dataset = 0
def LDA_classify(L1,Components):
    Xl = L1[:,:-1]
    yl = map(int,np.ravel(Wave[:,[4]]))
    lda = LinearDiscriminantAnalysis(n_components=Components)
    lda.fit(Xl,yl)
    print'-----------------------VALORES DEL LDA--------------------------------'
    print (lda.explained_variance_ratio_)
    print'----------------------------------------------------------------------'
    Xl = lda.fit_transform(Xl,yl)
    New_Datos = np.zeros((len(L1),Components+1))
    for i in range(len(L1)):
		for j in range(Components):
			New_Datos[i][j] = Xl[i][j]
		New_Datos[i][-1] = L1[i][-1]
    return New_Datos

ldax= LDA_classify(Wave, 3)
# print 'shape ldax', np.shape(ldax)
# print 'shape pcax', np.shape(pcax[:,-2:])
# ldax = np.concatenate((ldax,pcax[:,-2:]),axis=1)
# print 'LDA',ldax

# ------------------------------------------------------------------------------
#                               CLASIFICADOR SVM
# ------------------------------------------------------------------------------

max=0
min=1
degree=3
C=1
K="poly"

while C < 6:
    datawave = np.random.permutation(ldax)
    C1 = ldax[np.where(ldax[:,-1]==1),:]
    C2 = ldax[np.where(ldax[:,-1]==2),:]
    C3 = ldax[np.where(ldax[:,-1]==3),:]
    C4 = ldax[np.where(ldax[:,-1]==4),:]
    # print 'C1', np.shape(C1)
    # print 'C2', np.shape(C2)
    Xs = np.concatenate((C1[0][:,:],C2[0][:,:],C3[0][:,:],C4[0][:,:]),axis=0)
#    print 'SHAPE', np.shape(Xs)
    Xs = np.random.permutation(Xs)

    Ys = Xs[:,-1]

    print 'Creando clasificador con C:'+ str(C)
    clf = svm.SVC(kernel=K, C=C, degree=degree)
    valores = cross_val_score(clf,Xs[:,:-1],Ys, cv=4, scoring="f1_macro")

    for i in range(0,len(valores)):
        if valores[i] > max:
          max=valores[i]
        elif valores[i] < min:
          min=valores[i]
    print valores

    C = C + 1
print'-----------------------CLASIFICADOR SVM----------------------------------'
print "f1_macro", ((max+min)/2)*100
print'-------------------------------------------------------------------------'
#
#
# labels=ldax[:1000,-1]
# w0=np.where(labels==1)
# w1=np.where(labels==2)
# w2=np.where(labels==3)
# w3=np.where(labels==4)
# p=pl.plot(ldax[w0,0],ldax[w0,1],'bo')
# p=pl.plot(ldax[w1,0],ldax[w1,1],'ro')
# p=pl.plot(ldax[w2,0],ldax[w2,1],'go')
# p=pl.plot(ldax[w3,0],ldax[w3,1],'mo')
# pl.show()
