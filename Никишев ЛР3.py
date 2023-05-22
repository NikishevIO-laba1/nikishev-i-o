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


class nn:
    def __init__(self, m, hidden, k, initial_scale=0.1):
        """m - кол-во входных значений, k - кол-во классов, initial_scale - множитель изначальных случайных весов от 0 до 1"""
        self.W1 = np.random.random(size = (m+1, hidden))*initial_scale # веса
        self.W2 = np.random.random(size = (hidden+1, k))*initial_scale # веса
        self.k = k

    def predict(self, X):
        """X - входные данные массив размером (n, m)"""
        X = np.round(X/255)
        self.X = np.pad(X, pad_width=((0,0),(0,1)), constant_values=1)
        self.Yhat1 = sigmoid(np.dot(self.X, self.W1))
        self.Yhat1 = np.pad(self.Yhat1, pad_width=((0,0),(0,1)), constant_values=1)
        self.Yhat2 = softmax(np.dot(self.Yhat1, self.W2))
        return self.Yhat2
    
    def learn(self, x, y, learning_rate = 0.01):
        """x - объект обучающей выборки, y = класс"""
        y1 = np.zeros(shape = self.k)
        y1[y] = 1
        #стохастический градиентный спуск
        self.predict(x)
        delta2 = self.Yhat2 - y1 # вектор значений δ для параметров выходного слоя
        self.W2 -= learning_rate*np.dot(self.Yhat1.T, delta2)
        delta1 = (self.Yhat1 * (1- self.Yhat1)) * np.dot(delta2, self.W2.T)
        self.W1 -= learning_rate * np.dot(self.X.T, delta1[:,:-1])
    


import csv, matplotlib.pyplot as plt

def train(data = 'mnist_train.csv', rows = -1, hidden=20, learning_rate = 0.01, initial_scale=0.1):
    model = nn(m=784, hidden = hidden, k=10, initial_scale=initial_scale)
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

def test(rows = 3000, hidden=20, learning_rate=0.01, initial_scale = 0.1):
    model = train(rows = rows, hidden = hidden, learning_rate = learning_rate, initial_scale=initial_scale)
    print(f'Кол-во элементов: {rows}; Кол-во нейронов скрытого слоя: {hidden}; Скорость обучения - {learning_rate}')
    print(f'Точность модели - {accuracy(model = model)}\n')
    #confusion_matrix(model = model)
    #predict(model = model, image = load_image(image=50))

hidden = 200
learning_rate = 0.01
#for i in [10,100,500,1000,2500,5000,10000,25000,50000,100000]: test(rows=i, hidden=hidden, learning_rate=learning_rate)
model = train(rows = 60000, learning_rate=learning_rate, hidden=hidden)
print(accuracy(model=model))
confusion_matrix(model=model)
predict(model, load_image(50))
predict(model, load_image(51))
predict(model, load_image(52))
predict(model, load_image(53))