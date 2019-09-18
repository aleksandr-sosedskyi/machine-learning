from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

data = load_iris()

features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names
helpers = []

for t in range(3):
    if t == 0:
        c = 'r'
        marker = '>'
    elif t == 1:
        c = 'g'
        marker = 'o'
    elif t == 2:
        c = 'b'
        marker = 'x'
    helpers.append(plt.scatter(features[target == t, 0],
                               features[target == t, 1],
                               marker=marker,
                               c=c))

plt.legend(tuple(helpers),
           ('Iris Setosa', 'Iris Versicolour', 'Iris Virginica',),
           loc='best'
)
axes = plt.axes()
plt.title('Sepal length to sepal width')
axes.set_ylabel('Sepal length')
axes.set_xlabel('Sepal width')

plt.show()
plt.clf()
plt.cla()
plt.close()

plt.hist(features[:,2], label='Petal Length')
plt.title('Petal Length Histogram')
plt.xlabel('Petal length')
plt.ylabel('Distribution')
plt.legend()
plt.show()
