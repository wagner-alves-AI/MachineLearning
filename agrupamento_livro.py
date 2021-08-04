# -*- coding: utf-8 -*-
"""
Created on Tue May 18 18:35:53 2021

@author: wgnr2
"""

'''
exemplo 1

Importa o módulo que contém os conjuntos de dados

'''
from sklearn import datasets as ds

# carrega o conjunto iris de dados
iris = ds.load_iris()
print(iris.feature_names)

# exemplo 2
# importa a biblioteca pandas 
import pandas as pd

# cria um DataFrame a partir do objeto iris
df = pd.DataFrame(data=iris['data'],
                 columns=iris['feature_names'])

# adiciona ao DataFrame a coluna de atributos alvo
dfT = df.copy()
dfT['target'] = iris['target']
display(df), display(dfT)

# exemplo 3 - criar um conjunto de 100 instância e 4 atributos

from sklearn import datasets as ds
mb = ds.make_blobs(n_samples=100, n_features=4)
mb

# exemplo 4 
# importação do módulo Kmeans 

from sklearn.cluster import KMeans

# alg recebe o objeto do método kmeans aleatório básico

alg = KMeans(init='random')

# exemplo 5
# importação do módulo Kmeans
from sklearn.cluster import KMeans

# alg recebe o objeto do método Kmeans configurado
# para identificar 4 clusters. o valor 50 serve de base
# para o cálculo do ponto inicial de cada centroide.

alg = KMeans(n_clusters=4, random_state=50)

# exemplo 6
# Carrega libs e módulos necessários
from sklearn import datasets as ds
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pp

# distribuição aleatória: 100 objetos, 4 grupos, 2 características
# desvio padrão de 1.3
X, y = ds.make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.3)

# inicializa e treina um algoritmo K-means
km = KMeans(n_clusters=4, random_state=50)
km.fit(X)
prevs = km.predict(X)

# DataFrames para o conjunto de dados original com seu
# atributo alvo e outro quadro com o atributo alvo previsto
# a partir do K-Means
xDF = pd.DataFrame(data=X, columns=['Attr1', 'Attr2'])
pDF = xDF.copy()
xDF['Target'] = y
pDF['Target'] = prevs

# divisao da figura em duas colunas para plotar os dois
# conjuntos
fig, axes = pp.subplots(1, 2, figsize=(12,4))
sb.scatterplot(data=xDF, x='Attr1', y='Attr2', hue='Target',
              palette='rainbow', ax=axes[0]).set_title('Original')
sb.scatterplot(data=pDF, x='Attr1', y='Attr2', hue='Target',
               palette='rainbow', ax=axes[1]).set_title('Previsto')
# Exemplo 7
# importação das bibliotecas e seleção dos módulos a serem 
# utilizados
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pa
import seaborn as sb
import matplotlib.pyplot as pp
import numpy as np
# importação e separação dos dados
iDS = load_iris()
iData = iDS.data
iTarget = iDS.target
iTarget_names = iDS.target_names
iFeature_names = iDS.feature_names


# formatação dos dados em quadro da biblioteca pandas para 
# facilitar a visualização
iDF = pa.DataFrame(data=iData, columns=iFeature_names)
iDF_Target = iDF.copy()
iDF_Target['specie'] = iTarget_names[iTarget]
# visualizar a sobreposição das espécies entre todas as 
# combinações de pares de características
print(iDF)
print(iDF_Target)
sb.pairplot(data=iDF_Target, hue='specie')

# Exemplo 8
# seleção das duas características
iData_SL_PW = iData[:, [0,3]]
iDF_SL_PW = pa.DataFrame (data = iData_SL_PW , 
columns=[iFeature_names[0], iFeature_names[3]])
iDF_SL_PW_T = iDF_SL_PW.copy()
iDF_SL_PW_T['specie'] = iDF_Target['specie']
# distribuição após seleção dos atributos sem rótulos 
# definidos
prevFeatureNames = iDF_SL_PW_T.keys()
sb.scatterplot(data=iDF_SL_PW_T, x=prevFeatureNames[0], 
y=prevFeatureNames[1])

# Exemplo 9
# quantidade de clusters será a mesma da quantidade de 
# espécies do conjunto de dados
k = len(iTarget_names)
# configuração do algoritmo e treinamento
alg = KMeans(n_clusters=k, random_state=50)
alg.fit(X=iData_SL_PW)
# construção de novo quadro considerando os agrupamentos 
# previstos pelo K-Means
prevDataFrame = iDF_SL_PW.copy()
prevDataFrame['specie'] = alg.predict(iData_SL_PW)
prevDataFrame
sb.pairplot(data=prevDataFrame, hue='specie')

# Exemplo 10
translateDic = {0: 'virginica', 1: 'setosa', 2: 'versicolor' }
prevDataFrame['specie'] = [translateDic[s] for s in prevDataFrame['specie']]
fig, axes = pp.subplots(1, 2, figsize=(12,4))
fig1 = sb.scatterplot (data = iDF_SL_PW_T , 
x=prevFeatureNames[0], y=prevFeatureNames[1], hue='specie', 
ax=axes[0])
fig2 = sb.scatterplot(data = prevDataFrame , 
x=prevFeatureNames[0], y=prevFeatureNames[1], hue='specie', 
ax=axes[1])

# exemplo 11

len (prevDataFrame[iDF_SL_PW_T['specie']] != prevDataFrame['specie'])
cm = confusion_matrix(iDF_SL_PW_T['specie'], prevDataFrame['specie'])
'''cmtx = pa.DataFrame(
    cm, 
    index=('Real: ' + pa.DataFrame(iTarget_names)) [0],
    columns=('Prev: ' + pa.DataFrame(iTarget_names))[0]
)
cmtx, accuracy_score(iDF_SL_PW_T['specie'],
                    prevDataFrame['specie'])
'''
cmtx = pa.DataFrame(
 cm, 
 index=('Real: ' + pa.DataFrame(iTarget_names))
[0],
 columns=('Prev: ' + pa.DataFrame(iTarget_names))[0]
 )
cmtx , accuracy_score (iDF_SL_PW_T ['specie'] , 
prevDataFrame['specie'])