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
# def LDA_classify(L1,Components):
#     Xl = L1[:,:-1]
#     yl = map(int,np.ravel(Wave[:,[22]]))
#     # print 'yyyyyyyyyyyyyyyyyyyy ', yl
#     lda = LinearDiscriminantAnalysis(n_components=Components)
#     lda.fit(Xl,yl)
#     print'-----------------------VALORES DEL LDA--------------------------------'
#     print (lda.explained_variance_ratio_)
#     print'----------------------------------------------------------------------'
#     Xl = lda.fit_transform(Xl,yl)
#     New_Datos = np.zeros((len(L1),Components+1))
#     for i in range(len(L1)):
# 		for j in range(Components):
# 			New_Datos[i][j] = Xl[i][j]
# 		New_Datos[i][-1] = L1[i][-1]
#     return New_Datos
#
# # Dataset = loadtxt('AsignacionClases.csv', delimiter=',')
# Wave[0:,:22] = Wave[:,:22]-Wave[:,:22].mean(axis=0)
# Wave[0:,:22] = Wave[:,:22]/Wave[:,:22].var(axis=0)
# datawave = np.random.permutation(Wave)
#
# ldax=LDA_classify(datawave,4)
# print 'LDAX', ldax

# ------------------------------------------------------------------------------
#                               CLASIFICADOR SVM
# ------------------------------------------------------------------------------

max=0
min=1
degree=3
C=1
K="poly"

while C < 3:

    # data = array(ldax)
    Xs = Wave[:9000,:20]
    # Ys = ldax[:,-1]
    Ys = map(int,Wave[:9000,-1])
# Xs = ldax[:200,3:]
    # print 'x',Xs
# Ys = ldax[:200,3]
    # print 'Y', Ys
    print 'Creando clasificador con C:'+ str(C)
    clf = svm.SVC(kernel=K, C=C, degree=degree)
    valores = cross_val_score(clf,Xs,Ys, cv=4, scoring="f1_macro")

    for i in range(0,len(valores)):
        if valores[i] > max:
          max=valores[i]
        elif valores[i] < min:
          min=valores[i]
    print valores

    C = C + 1
print'-----------------------CLASIFICADOR SVM----------------------------------'
# print "Maximo", max
# print "Minimo", min
print "Promedio", ((max+min)/2)*100
print'-------------------------------------------------------------------------'


# print'-----------------------MATRIZ DE CONFUSION--------------------------------'
# cm = confusion_matrix(valores[0:,:], resultado)
#
# print(cm)
# acierto = 0
# error = 0
# for i in range(0,len(valores)):
#     if valores[i,22]==resultado[i]:
#         acierto=acierto+1
#     elif valores[i,22]!=resultado[i]:
#         error=error+1
#
# promedio = float(acierto) / (acierto + error)
# print "Promedio", promedio*100, "%"
#
# # Grafica de Matriz de Confusion
# plt.matshow(cm)
# plt.title('Matriz de Confusion')
# plt.colorbar()
# plt.ylabel('Valores Reales')
# plt.xlabel('Valor Predicho')
# plt.show()
# print'-------------------------------------------------------------------------'
# ------------------------------------------------------------------------------
#                                    MLP
# ------------------------------------------------------------------------------

# Datos = ldax
# random.shuffle(Datos)
#
# #SEPARO BASE DE DATOS 50%
# train_number=len(Datos)/2
# error=0
# correcto=0
#
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6), random_state=1)
#
# #NORMALIZO DATOS
#
# Datos[0:,:21] = Datos[:,:21]-Datos[:,:21].mean(axis=0)
# Datos[0:,:21] = Datos[:,:21]/Datos[:,:21].var(axis=0)
#
# #CLASIFICACION
#
# repeat = 10
# for i in range(0,repeat):
#     CopiaBD = Datos
#     aleacion = np.random.permutation(CopiaBD)
#     print 'ALEACION',aleacion
#     datos_entrenamiento = np.array(aleacion[:train_number,:5])
#     datos_entrenamiento2= np.array(aleacion[:train_number,4])
#     datos_prueba = np.array(aleacion[train_number+1:,:5])
#     datos_prueba2 = np.array(aleacion[train_number+1:,4])
#
#     #MODELO MLPC
#
#     clf.fit (datos_entrenamiento[0:,0:5],datos_entrenamiento2)
#
#     #PREDICE
#     datos_prueba_p = clf.predict(datos_prueba)
#     MatrixC = np.zeros((2,2))
#
#     #MATRIXC
#     for i in range(0, len(datos_prueba2)):
#         if datos_prueba2[i] == datos_prueba_p[i]:
#             correcto = correcto + 1
#             if datos_prueba2[i] == 0:
#                 MatrixC[0,0] = MatrixC[0,0] + 1
#             elif datos_prueba2[i] == 1:
#                 MatrixC[1,1] = MatrixC[1,1] + 1
#
#         else:
#             error = error + 1
#             if datos_prueba2[i] == 0:
#                 MatrixC[1,0] = MatrixC[1,0] + 1
#             elif datos_prueba2[i] == 1:
#                 MatrixC[0,1] = MatrixC[0,1] + 1
#
#     aciertos = 0
#     errores = 0
#
# print MatrixC
# Total = correcto + error
#
# Avg = correcto/ float(Total)
# Avg = Avg * 100
#
# #print 'Total de datos: ', len(Datos)
# print "Promedio: ", Avg,"%"


# ------------------------------------------------------------------------------
#                  GRAFICA DE DISCRETE WAVELET TRANSFORM
# ------------------------------------------------------------------------------

# x = loadtxt("AsignacionClases.csv", delimiter=",")
# x = ldax[:,:4]
# cont = 0
# cnt = cont + 1
# pl.figure(2)
# pl.clf()
# pl.plot(x, label='signal')
# pl.xlabel('time (seconds)')
# pl.grid(True)
# pl.axis('tight')
# pl.legend(loc='upper left')
#
# pl.show()
