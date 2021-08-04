# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 03:22:46 2021

@author: https://www.youtube.com/watch?v=ZfIb__rg2As&list=PLZj-vsMJRNhprMuIaE6HXmOkHh0NEEMtL&index=4
"""

import numpy as np

# define o nr de epocas e o nr de amostras (q)
numEpocas = 70000
q = 6

# Atributos
peso = np.array([113, 122, 107, 98, 115, 120])
pH = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])

# Bias
bias = 1

# Entrada do perceptron

X = np.vstack((peso, pH))
Y = np.array([-1, 1, -1, -1, 1, 1])

# Taxa de aprendizado
eta = 0.1

# define o vetor de pesos
W = np.zeros([1,3])         # Duas entradas + o Bias

# Array para armazenas os erros
e = np.zeros(6)

def funcaoAtivacao(valor):
    # A funcao de ativacao Degrau bipolar
    if valor < 0.0:
        return(-1)
    else:
        return(1)
    
for j in range(numEpocas): 
    for k in range(q):
        # Insere o bias no vetor de entrada
        Xb = np.hstack((bias, X[:,k]))
        
        # Calcula o campo induzido
        V = np.dot(W, Xb)       #Equacao (5)
        
        # Calcula a saída do perceptron
        Yr = funcaoAtivacao(V)  #Equacao (6)
        
        # Calcula o erro : e = (Y - Yr)
        e[k] = Y[k] - Yr
        
        # Treinamento do perceptron
        W = W + eta*e[k]*Xb

print('Vetor de erros (e) = ' + str(e)) 