from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = datasets.load_iris()

cluster = KMeans(n_clusters = 3)
cluster.fit(iris.data)

previsoes = cluster.labels_
centroides = cluster.cluster_centers_

resultados = confusion_matrix(iris.target, previsoes)

plt.scatter(iris.data[previsoes == 0, 0], iris.data[previsoes == 0, 3], 
            c = 'green', label = 'Setosa')
plt.scatter(iris.data[previsoes == 1, 0], iris.data[previsoes == 1, 3], 
            c = 'red', label = 'Versicolor')
plt.scatter(iris.data[previsoes == 2, 0], iris.data[previsoes == 2, 3], 
            c = 'blue', label = 'Virgica')
plt.legend()