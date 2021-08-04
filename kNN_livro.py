# -*- coding: utf-8 -*-
"""
Created on Tue May 25 02:01:37 2021

@author: wgnr2
"""

# Exemplo 1 - Carregadndo e codificando os dados
from sklearn import preprocessing as pe
import pandas as pa

data = [

        [ 'Ensolarado', 28, 'Alta', 'Fraco', 'Não' ],
        [ 'Ensolarado', 31, 'Alta', 'Forte', 'Não' ],
        [ 'Nublado', 30, 'Alta', 'Fraco', 'Sim' ],
        [ 'Chuvoso', 23, 'Alta', 'Fraco', 'Sim' ],
        [ 'Chuvoso', 15, 'Normal', 'Fraco', 'Sim' ],
        [ 'Chuvoso', 15, 'Normal', 'Forte', 'Não' ],
        [ 'Nublado', 11, 'Normal', 'Forte', 'Sim' ],
        [ 'Ensolarado', 21, 'Alta', 'Fraco', 'Não' ],
        [ 'Ensolarado', 12, 'Normal', 'Fraco', 'Sim' ] 
        
        ]

df = pa.DataFrame(data, columns=['Clima', 'Temperatura', 'Umidade', 
                                 'Vento', 'Jogar'])

def Encode(lista):
    enc = pe.LabelEncoder()
    enc.fit(lista)
    nova_lista = enc.transform(lista)
    return (enc, nova_lista)

peClima, df['Clima'] = Encode(df['Clima'])
peUmidade, df['Umidade'] = Encode(df['Umidade'])
peVento, df['Vento'] = Encode(df['Vento'])
peJogar, df['Jogar'] = Encode(df['Jogar'])

display(df)

# Exemplo 2 - Seleção de atributos e separação entre
# previsores/alvo
import numpy as np

X = np.array(df[['Clima', 'Temperatura']])
y = np.array(df['Jogar'])
display(X)
display(y)

# Exemplo 3 - Treinamento de um modelo kNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X,y)

# Exemplo 4 - Previsao dos rótulos
import seaborn as sb

y_prevs = knn.predict(X)

df_prev = pa.DataFrame(X, columns=['Clima', 'Temperatura'])
df_prev['Jogar'] = y_prevs
df_prev['Jogar'] = peJogar.inverse_transform(df_prev['Jogar'])

sb.pairplot(df_prev, hue='Jogar')

# exemplo 5 - Comparação entre os rótulos previstos e os
# originais
yy = y_prevs == y

df_prev_check = pa.DataFrame(X, columns=['Clima', 'Temperatura'])
df_prev_check['Rotulo'] = yy

sb.pairplot(df_prev_check, hue='Rotulo')

# Exemplo 6 - Métrica de acurácia
from sklearn.metrics import accuracy_score

acc = accuracy_score(y, y_prevs)
display('Acuracia: {0:0.2f}%'.format(acc*100))

# Exemplo 7 - Matriz de confusão
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_prevs)
cm_df = pa.DataFrame(cm, index=['Nao - Real', 'Sim - Real'],
             columns=['Nao - previsto', 'Sim - Previsto'])

# Exemplo 8 – Varredura de parâmetros
neighbors = range(1, 9, 1)
weights = {'uniform': [], 'distance': []}
for w in weights:
    accs = [0]
    sens = [0]
    esps = [0]
 
    for n in neighbors:
        model = KNeighborsClassifier(n_neighbors=n, 
                                     weights=w)
        model.fit(X, y) 
        y2 = model.predict(X)
 
    VN, FP, FN, VP = confusion_matrix(y, y2).ravel()
    acc = accuracy_score(y, y2)*100
    sen = VP/(VP+FN)*100
    esp = VN/(VN+FP)*100
    accs.append(acc)
    sens.append(sen)
    esps.append(esp) 
    weights[w] = [accs, sens, esps]

import matplotlib.pyplot as pyp
fig, axs = pyp.subplots(1, 2, figsize=(12,4))
for i, d in enumerate(weights.items()):
    for dd in d[1]:
        axs[i].plot(dd)
        axs[i].set_title('Pesos (weights): {}'.format(d[0]))
        axs[i].set_xlabel('k-Vizinhos (n _ neighbors)')
        axs[i].set_ylabel('Taxa (%)')
        axs[i].legend(['Acurácia', 'Sensibilidade',
                             'Especificidade'])