from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

iris_teste = iris.data[0,:]
iris_teste_classe = iris.target[0]

X = iris.data[1:150,:]
y = iris.target[1:150]

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y)

previsao = knn.predict(iris_teste.reshape(1,-1))

