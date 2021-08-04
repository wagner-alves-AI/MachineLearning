# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:48:31 2021

@author: wgnr2
"""

# Este exemplo carrega a base Wine da UCI, treina um classificador SVM
# usando holdout e outro usando validaÃ§Ã£o cruzada com 10 pastas. 

# Importa bibliotecas necessÃ¡rias 
import numpy as np
import urllib
from sklearn.svm import SVC
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#from sklearn.model_selection import StratifiedShuffleSplit
# Carrega uma base de dados do UCI
# Exemplo carrega a base Wine
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
raw_data = urllib.request.urlopen(url)

# Carrega arquivo como uma matriz
dataset = np.loadtxt(raw_data, delimiter=",")

# Imprime quantide de instÃ¢ncias e atributos da base
print(dataset.shape)

# Coloca em X os 13 atributos de entrada e em y as classes
# Observe que na base Wine a classe Ã© primeiro atributo 
X = dataset[:,1:13]
y = dataset[:,0]

# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42, stratify=y)

# Treina o classificador

#DefiniÃ§Ã£o dos parÃ¢metros a serem avaliados no ajuste fino do SVM
parameters = [{'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['linear']},{'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']}]

clfa = SVC()
clfa = GridSearchCV(clfa, parameters, scoring = 'accuracy', cv=10)
clfa = clfa.fit(X_train, y_train)
print(clfa.best_params_)

# testa usando a base de testes
predicted=clfa.predict(X_test)

# calcula a acurÃ¡cia na base de teste
score=clfa.score(X_test, y_test)

# calcula a matriz de confusÃ£o
matrix = confusion_matrix(y_test, predicted)

# apresenta os resultados
print("Accuracy = %.2f " % score)
print("Confusion Matrix:")
print(matrix)

# EXEMPLO USANDO VALIDAÃ‡ÃƒO CRUZADA


clfb = SVC()
clfb = GridSearchCV(clfb, parameters, scoring = 'accuracy', cv=10)
folds=10
result = model_selection.cross_val_score(clfb, X, y, cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())

# matriz de confusÃ£o da validaÃ§Ã£o cruzada
Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)
cm=confusion_matrix(y, Z)
print("Confusion Matrix:")
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, predicted))





