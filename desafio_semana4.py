# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 01:26:32 2021

@author: wgnr2
"""

from six import StringIO 
from sklearn.tree import export_graphviz 
from IPython.display import Image
import pydotplus

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

X, y, target, _, features, _, __ = load_breast_cancer().values()

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=1)

model = DecisionTreeClassifier(max_depth=5, random_state=1, criterion='entropy')

model.fit(X_treino, y_treino)

y_prev = model.predict(X_teste)

acc = '{0:2.1f}'.format(accuracy_score(y_teste, y_prev)*100)

fscore_holdout = f1_score(y_teste, y_prev)
print("F1-score: %.2f" % fscore_holdout)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_teste, y_prev)
matrix

model2 = DecisionTreeClassifier(max_depth=5, criterion='entropy')
folds=5
model2.fit(X_treino, y_treino)
resultado = model_selection.cross_val_score(model2, X, y, cv=folds)

y_prev2 = model2.predict(X_teste)


print("\nResultados baseados em Validacao Cruzada")
print("Qtde folds: %d:" % folds)
print("Taxa de Acerto: %.2f" % resultado.mean())
print("Desvio padrao: %.2f" % resultado.std())

#precisão
pres = model_selection.cross_val_score(model2, X, y, cv=folds, 
                                         scoring='precision')
print("\nPrecisao %d folds:" % folds)
print("Mean precision: %.2f" % pres.mean())
print("Mean Std: %.2f" % pres.std())

#F1-score
fscore = model_selection.cross_val_score(model2, X, y, cv=folds, 
                                         scoring='f1')
print("\nF1-score %d folds:" % folds)
print("Mean F1-score: %.2f" % fscore.mean())
print("Mean Std: %.2f" % fscore.std())

print("\nA arvore gerada no experimento baseado em Cross Validation =")
dot_data = StringIO()
export_graphviz(model2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
im=Image(graph.create_png())
display(im)
