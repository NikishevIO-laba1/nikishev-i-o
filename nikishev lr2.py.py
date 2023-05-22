# Никишев И.О. 224-321 ЛР2  Распознавание изображений с помощью персептронагоритмов
import numpy as np,math
np.random.seed(1)

def sigmoid(v):
    return 1/(1+(math.e**(-v)))

def heavyside(v):
    v[v>0]=1
    v[v<0]=0
    return v

def softmax(v):
    return np.exp(v-np.max(v))/np.sum((np.exp(v-np.max(v))))


class perceptron:
    def __init__(self, m, k, initial_scale=0.1):
        """m - кол-во входных значений, k - кол-во классов, initial_scale - множитель изначальных случайных весов от 0 до 1"""
        self.W = np.random.random(size = (m+1, k))*initial_scale # веса
        self.k = k

    def predict(self, X, activation_func = heavyside):
        """X - входные данные массив размером (n, m)"""
        X = np.round(X/255)
        self.X = np.pad(X, pad_width=((0,0),(0,1)), constant_values=1)
        Yhat = np.dot(self.X, self.W)
        return activation_func(Yhat)
    
    def learn(self, x, y, learning_rate = 0.01):
        """x - объект обучающей выборки, y = класс"""
        Yhat = self.predict(x)
        y1 = np.zeros(shape = self.k)
        y1[y] = 1
        e = y1 - Yhat # вектор ошибок  
        D = learning_rate*np.dot(self.X.T, e) # матрица значений корректировок весов. Корректировка пропорциональна произведению сигнала ошибки на входной сигнал, его вызвавший
        self.W += D
    


import csv, matplotlib.pyplot as plt

def train(data = 'mnist_train.csv', rows = -1, learning_rate = 0.01, initial_scale=0.1):
    model = perceptron(m=784, k=10, initial_scale=initial_scale)
    with open(data) as f:
        reader = list(csv.reader(f, delimiter=','))
        while True:
            for row in reader: 
                model.learn(x=np.array([row[1:]], dtype=int), y=int(row[0]), learning_rate=learning_rate)
                rows-=1
                #if rows%1000 == 0: print(rows)
                if rows==0: 
                    rows-=1
                    break
            if rows<0: break
        return model

def accuracy(data = 'mnist_test.csv', model = None):
    with open(data) as f:
        reader = csv.reader(f, delimiter=',')
        data = np.array(list(reader), dtype=int)
        inputs = data[:, 1:]
        classes = data[:, 0]
        Yhat = np.argmax(model.predict(inputs), axis=1)
        accuracy = np.sum(Yhat == classes)/len(classes)
        return accuracy

def confusion_matrix(data = 'mnist_test.csv', model = None):
    with open(data) as f:
        reader = csv.reader(f, delimiter=',')
        data = np.array(list(reader), dtype=int)
        inputs = data[:, 1:]
        classes = data[:, 0]
        Yhat = np.argmax(model.predict(inputs), axis=1)
        cm = np.zeros((10, 10), dtype=int)
        for i in range(len(classes)):
            cm[classes[i]][Yhat[i]] += 1
        plt.imshow(cm, cmap="Blues")
        plt.xlabel("предсказано")
        plt.ylabel("реальные значения")
        plt.colorbar()
        for i in range(10):
            for j in range(10):
                plt.text(x=j, y=i, s=cm[i, j], ha="center", va="center", color="white")
        plt.show()

def predict(model, image):
    plt.imshow(np.round(image.reshape((28, 28))/255), cmap='Greys', interpolation='nearest')
    prediction = np.argmax(model.predict(np.array([image], dtype=int)))
    plt.title(f'Предсказание модели - {prediction}')
    plt.show()

def load_image(data = 'mnist_test.csv',image=1):
    with open(data) as f:
        reader = csv.reader(f, delimiter=',')
        return np.array(list(reader)[image][1:], dtype=int)

def test(rows = 3000, learning_rate=0.01, initial_scale = 0.1):
    model = train(rows = rows, learning_rate = learning_rate, initial_scale=initial_scale)
    print(f'Кол-во элементов: {rows}; Скорость обучения - {learning_rate}')
    print(f'Точность модели - {accuracy(model = model)}\n')
    #confusion_matrix(model = model)
    #   predict(model = model, image = load_image(image=50))

test(rows = 10)
test(rows = 100)
test(rows = 500)
test(rows = 1000)
test(rows = 2500)
test(rows = 5000)
test(rows = 10000)
test(rows = 25000) 
test(rows = 50000) 
test(rows = 100000)