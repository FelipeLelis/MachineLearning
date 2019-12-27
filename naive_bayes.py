import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('insurance.csv')
base = base.drop(columns = ['Unnamed: 0'])
base.Accident.unique()

X = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values
y = base.iloc[:, 7].values

labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
X[:,1] = labelencoder.fit_transform(X[:,1])
X[:,2] = labelencoder.fit_transform(X[:,2])
X[:,3] = labelencoder.fit_transform(X[:,3])
X[:,4] = labelencoder.fit_transform(X[:,4])
X[:,5] = labelencoder.fit_transform(X[:,5])
X[:,6] = labelencoder.fit_transform(X[:,6])
X[:,7] = labelencoder.fit_transform(X[:,7])
X[:,8] = labelencoder.fit_transform(X[:,8])
X[:,9] = labelencoder.fit_transform(X[:,9])
X[:,10] = labelencoder.fit_transform(X[:,10])
X[:,11] = labelencoder.fit_transform(X[:,11])
X[:,12] = labelencoder.fit_transform(X[:,12])
X[:,13] = labelencoder.fit_transform(X[:,13])
X[:,14] = labelencoder.fit_transform(X[:,14])
X[:,15] = labelencoder.fit_transform(X[:,15])
X[:,16] = labelencoder.fit_transform(X[:,16])
X[:,17] = labelencoder.fit_transform(X[:,17])
X[:,18] = labelencoder.fit_transform(X[:,18])
X[:,19] = labelencoder.fit_transform(X[:,19])
X[:,20] = labelencoder.fit_transform(X[:,20])
X[:,21] = labelencoder.fit_transform(X[:,21])
X[:,22] = labelencoder.fit_transform(X[:,22])
X[:,23] = labelencoder.fit_transform(X[:,23])
X[:,24] = labelencoder.fit_transform(X[:,24])
X[:,25] = labelencoder.fit_transform(X[:,25])

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
modelo = GaussianNB()
modelo.fit(X_treinamento, y_treinamento)

previsoes = modelo.predict(X_teste)

accuracy_score(y_teste, previsoes)

confusao = ConfusionMatrix(modelo, classes=['None','Severe','Mild','Moderate'])
confusao.fit(X_treinamento, y_treinamento)
confusao.score(X_teste, y_teste)
confusao.poof()