from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Инициализация объектов DataFrame
iris = read_csv('iris/iris.data')
wine = read_csv('wine/wine.data')
zoo = read_csv('zoo/zoo.data')

# Процедура случайного перемешивания и деления выборки на тестовую и обучающую
wine_sample = DataFrame.sample(wine, frac=1)
wine_train, wine_test = train_test_split(wine_sample, test_size=0.2)

# Процедура назначений номеров классов для символьных значений
label = LabelEncoder()
dicts = {}
label.fit(iris['Iris-setosa'].drop_duplicates())
iris['Iris-setosa'] = label.transform(iris['Iris-setosa'])















