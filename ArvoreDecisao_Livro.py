# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 02:33:01 2021

@author: wgnr2
"""

# Parte 1 - Classificador
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

# Parte 2 - Treinamento com Iris
from sklearn.datasets import load_iris

# Carrega os dados separando em previsores (X) e alvo (y)
X, y = load_iris(return_X_y=True)

# treina o modelo
model.fit(X=X, y=y)

# Parte 3 - Previsão, DataFrames e exibição de gráficos
# Bibliotecas necessárias
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pyp

# Dicionário que mapeará os rótulos numéricos para as espécies
dic = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# DF do conjuntos original
df = pd.DataFrame(X, columns=['Comprimento das sepalas',
                              'Largura das sepalas',
                              'Comprimento das petalas',
                              'Largura das petalas'])
df['Especie'] = y
df['Especie'] = df['Especie'].apply(lambda i: dic[i])

# Previsao dos rotulos a partir dos atributos previsores
y_prevs = model.predict(X)

# DataFrame com os rotulos previstos
df_prevs = df.copy()
df_prevs['Especie'] = y_prevs
df_prevs['Especie'] = df_prevs['Especie'].apply(lambda i: dic[i])

# Plotagens dos dados originais e previstos com matplotlib
fig, (ax1,ax2) = pyp.subplots(1,2, figsize=(10,4),sharex=False)


ax1.scatter(x=df['Comprimento das petalas'],
               y=df['Comprimento das sepalas'])
ax1.set_title('a) Rótulos originais')
ax2.scatter(x=df_prevs['Comprimento das petalas'],
               y=df_prevs['Comprimento das sepalas'])
ax2.set_title('b) Rótulos previstos')

# plotagem com seaborn

fig, axes = pyp.subplots(1, 2, figsize=(10,4))
sb.scatterplot( x=df['Comprimento das petalas'], 
                y=df['Comprimento das sepalas'], 
                hue=df['Especie'], ax=axes[0])
sb.scatterplot( x=df_prevs['Comprimento das petalas'], 
                y=df_prevs['Comprimento das sepalas'], 
                hue=df_prevs['Especie'], ax=axes[1])
axes[0].set_title('a) Rótulos originais.')
axes[1].set_title('b) Rótulos previstos.')

# Parte 4 - Plot tree
import sklearn.tree

#cria uma area de gráfico de tamanho específico
fig, ax = pyp.subplots(figsize=(16, 8))

'''plota a árvore preenchida pela cor da classe que predomina
no nó
'''
arvore = sklearn.tree.plot_tree(model, max_depth=1000, fontsize=10,
                                filled=True, ax=ax)
