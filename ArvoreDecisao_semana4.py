# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:33:54 2021

@author: wgnr2
"""

# Este exemplo carrega a base Wine da UCI, treina uma Arvore de decisao usando 
# holdout e outra usando validacao cruzada com 10 pastas. 

# Importa bibliotecas necessarias 

from sklearn.datasets import load_boston 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn import model_selection 
from sklearn.model_selection import train_test_split 
from six import StringIO 
from sklearn.tree import export_graphviz 
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.metrics import f1_score
import pydotplus


# Carregar base Cancer
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# Carrega arquivo como uma matriz
X, y = load_breast_cancer(return_X_y=True)

import collections
a = collections.Counter(y)


# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)


# declara o classificador
clfa = DecisionTreeClassifier(criterion='gini', max_depth=3)

# treina o classificador
clfa = clfa.fit(X_train, y_train)

# testa usando a base de testes
predicted=clfa.predict(X_test)

# calcula a acuracia na base de teste (taxa de acerto)
score=clfa.score(X_test, y_test)
score

fscore_holdout = f1_score(y_test,predicted)
print("F1-score: %.2f" % fscore_holdout)


# calcula a matriz de confusao
matrix = confusion_matrix(y_test, predicted)

# apresenta os resultados
print("\nResultados baseados em Holdout 70/30")
print("Taxa de acerto = %.2f " % score)
print("Matriz de confusao:")
print(matrix)



# EXEMPLO USANDO VALIDACAO CRUZADA
clfb = DecisionTreeClassifier(max_depth=3)
folds=5
result = model_selection.cross_val_score(clfb, X, y, cv=folds)

print("\nResultados baseados em Validacao Cruzada")
print("Qtde folds: %d:" % folds)
print("Taxa de Acerto: %.2f" % result.mean())
print("Desvio padrao: %.2f" % result.std())

# matriz de confusÃ£o da validacao cruzada
Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)
cm=confusion_matrix(y, Z)
print("Matriz de confusao:")
print(cm)


# 
score=clfa.score(X_test, y_test)
score*100
print("Acuracia = %.2f " % score)
print("Confusion Matrix:")
print(matrix)

#precisão
pres = model_selection.cross_val_score(clfb, X, y, cv=folds, 
                                         scoring='precision')
print("\nCross Validation Results %d folds:" % folds)
print("Mean precision: %.2f" % pres.mean())
print("Mean Std: %.2f" % result.std())

#F1-score
fscore = model_selection.cross_val_score(clfb, X, y, cv=folds, 
                                         scoring='f1')
print("\nCross Validation Results %d folds:" % folds)
print("Mean F1-score: %.2f" % fscore.mean())
print("Mean Std: %.2f" % result.std())

# matriz de confusão da matriz de confusão
Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)
cm=confusion_matrix(y, Z)
print("Confusion Matrix:")
print(cm)

#imprime a arvore gerada
print("\nA arvore gerada no experimento baseado em Holdout")
dot_data = StringIO()
export_graphviz(clfa, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
im=Image(graph.create_png())
display(im)
