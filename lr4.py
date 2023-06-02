#ФИО Никишев Иван Олегович nkshv2@gmail.com, учебный курс https://online.mospolytech.ru/mod/assign/view.php?id=281548, год 2023

import random, numpy as np, math
from functools import partial
seed = 0
random.seed(seed)
np.random.seed(seed)


def sigmoid(v):
    return 1/(1+(math.e**(-v)))

def heavyside(v):
    v[v>0]=1
    v[v<0]=0
    return v

def softmax(v):
    return np.exp(v-np.max(v))/np.sum((np.exp(v-np.max(v))))


class perceptron_ga:
    def __init__(self, m, k, initial_scale=1):
        """m - кол-во входных значений, k - кол-во классов, initial_scale - множитель изначальных случайных весов от 0 до 1"""
        self.m = m
        self.W = np.random.uniform(-1, 1, size = (m+1, k))*initial_scale # веса
        self.W_flat = self.W.ravel()
        self.k = k

    def predict(self, X, activation_func = heavyside):
        """X - входные данные массив размером (n, m)"""
        X = np.round(X/255)
        self.X = np.pad(X, pad_width=((0,0),(0,1)), constant_values=1)
        Yhat = np.dot(self.X, self.W)
        return activation_func(Yhat)
    
    # def learn(self, x, y, learning_rate = 0.01):
    #     """x - объект обучающей выборки, y = класс"""
    #     Yhat = self.predict(x)
    #     y1 = np.zeros(shape = self.k)
    #     y1[y] = 1
    #     e = y1 - Yhat # вектор ошибок  
    #     D = learning_rate*np.dot(self.X.T, e) # матрица значений корректировок весов. Корректировка пропорциональна произведению сигнала ошибки на входной сигнал, его вызвавший
    #     self.W += D
    
    # def fitness(self, data, y):
    #     Yhat = np.argmax(self.predict(data), axis=1)
    #     #Yhat = self.predict(data)
    #     return np.sum(Yhat == y)/len(y)

    def error(self, data, y):
        Yhat = self.predict(data).T[0]
        return np.sum(y == Yhat)/len(y)
        
    def confusion_matrix(self, data, y):
        Yhat = self.predict(data).T[0]
        return [[((Yhat == 0) & (y == 0)).sum(), ((Yhat == 1) & (y == 0)).sum()], [((Yhat == 0) & (y == 1)).sum(), ((Yhat == 1) & (y == 1)).sum()]]

    def fitness(self, data, y):
        Yhat = self.predict(data).T[0]
        e = (np.sum(y==Yhat))/len(y)
        return e
    
    def mutate_norm(self, sigma = 0.1):
        """К случайному весу прибавляется случайное значение из нормального распределения"""
        index = random.randrange(0, len(self.W_flat))
        self.W_flat[index] += random.gauss(0, sigma)

    def copy(self):
        copy = perceptron_ga(self.m, self.k, initial_scale=1)
        copy.W = self.W.copy()
        copy.W_flat = copy.W.ravel()
        return copy

def cross_swap_random(nn1: perceptron_ga, nn2: perceptron_ga, p = 0.5):
    """p% случайных весов меняются местами"""
    indices = np.random.choice(len(nn1.W_flat), size = int(len(nn1.W_flat)*p), replace = False)
    nn1.W_flat[indices], nn2.W_flat[indices] = nn2.W_flat[indices], nn1.W_flat[indices]
    return nn1,nn2

def cross_cut(nn1: perceptron_ga, nn2: perceptron_ga):
    """Слева от 1 особи справа от 2"""
    index = random.randrange(1, len(nn1.W_flat)-1)
    nn1.W_flat[index:], nn2.W_flat[index:] = nn2.W_flat[index:], nn1.W_flat[index:]
    nn1.W_flat[:index], nn2.W_flat[:index] = nn2.W_flat[:index], nn1.W_flat[:index]
    return nn1, nn2

def sel_roulette(population, fitness, num):
    """приспособленность особи является вероятностью её выбора"""
    pool=[]
    if np.min(fitness)<0: fitness -= np.min(fitness)
    # print(fitness)
    indexes = np.random.choice(a=len(fitness), size=num,replace=False,p= fitness/np.sum(fitness))
    for i in indexes: pool.append(population[i])
    return pool

def sel_tournament(population, fitness, num):
    """ все особи разбиваются на группы и выбирается наилучшая из каждой"""
    group_size = int(len(population)//num)
    groups=[]
    population = list(zip(population, fitness))
    while len(population)>=group_size:
        group=[]
        for i in range(group_size):
            ind = random.randrange(0, len(population))
            group.append(population[ind])
            population.pop(ind)
        groups.append(group)
    for i in range(len(groups)):
        groups[i] = max(groups[i], key = lambda i: i[1])[0]
    return groups

def sel_rank(population, fitness, num):
    """номер особи в отсортированном по приспособленности списке всех особей является вероятностью её выбора"""
    population = sorted(population, key = lambda x: fitness[population.index(x)])
    fitness = list(range(len(fitness)))
    return sel_roulette(population, fitness, num)


def chromosome_nn():
    return perceptron_ga(m=784, k=1)

class ga:
    def __init__(self, selection, crossover, chromosome, population_size = 200, pool_size = 25, crossover_p = 0.7, mutation_p = 0.01, first_gen = 5):
        self.selection = selection
        self.crossover = crossover
        self.chromosome = chromosome
        self.population_size = population_size
        self.pool_size = pool_size
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p

        self.population = [chromosome() for i in range(int(self.population_size*first_gen))]
        self.log_max_fitness =[]
        self.log_min_fitness =[]
        self.log_avg_fitness =[]

        self.gen=0
     
    def advance(self, data, y):
        self.fitness = [i.fitness(data, y) for i in self.population]
        pool = self.selection(self.population, self.fitness, num =self.pool_size)
        pool_copy = pool[:]
        self.population=[self.population[self.fitness.index(max(self.fitness))]]
        i=1
        while i < self.population_size:
            if random.uniform(0,1)<self.crossover_p:
                # СКРЕЩИВАНИЕ
                selected_inds = random.sample(pool, 2)
                crossover_inds = self.crossover(*[i.copy() for i in selected_inds])
                self.population.extend(crossover_inds)
                i+=len(crossover_inds)
            else:
                try:
                    ind = random.choice(pool_copy)
                    pool_copy.remove(ind)
                except:
                    ind = random.choice(self.population).copy()
                    ind.mutate_norm()
                self.population.append(ind)
                i+=1

        for i in range(1, len(self.population)):
            if random.uniform(0,1)<self.mutation_p:
                # МУТАЦИЯ
                self.population[i].mutate_norm()
        self.gen+=1
    
    def log_fitness(self):
        self.log_max_fitness.append(max(self.fitness))
        self.log_min_fitness.append(min(self.fitness))
        self.log_avg_fitness.append(np.mean(self.fitness))
        print(f'Поколение: {self.gen}, макс. приспособленность: {self.log_max_fitness[-1]}, мин: {self.log_min_fitness[-1]}, средняя: {self.log_avg_fitness[-1]}')

    def best(self):
        return self.population[self.fitness.index(max(self.fitness))]

                
import csv
def get_data(data = 'mnist_train.csv', rows = np.s_[:], classes = ('0', '1')):
    with open(data) as f:
        reader = list(csv.reader(f, delimiter=','))
        data = np.array([i for i in list(reader) if i[0] in classes], dtype=int)[rows]
        inputs = data[:, 1:]
        classes = data[:, 0]
        return inputs, classes

def accuracy(data = 'mnist_test.csv', model = None):
    data, y = get_data(data = data, rows = np.s_[:], classes = ('0', '1'))
    return model.error(data, y)

from matplotlib import pyplot as plt
def confusion_matrix(data = 'mnist_test.csv', model = None):
    data, y = get_data(data = data, rows = np.s_[:], classes = ('0', '1'))
    cm = model.confusion_matrix(data, y)
    print(cm)
    plt.imshow(cm, cmap="Blues")
    plt.xlabel("предсказано")
    plt.ylabel("реальные значения")
    a = plt.colorbar()
    for i in range(2):
        for j in range(2):
            import matplotlib.patheffects as pe
            plt.text(x=j, y=i, s=cm[i][j], ha="center", va="center", color="white", size=40, path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.show()

cross_swap_random_05 = partial(cross_swap_random, p = 0.5)

model = ga(selection = sel_tournament, crossover = cross_swap_random_05, chromosome = chromosome_nn, population_size = 50, pool_size = 15, crossover_p = 0.75, mutation_p = 0.05, first_gen = 1)

for i in range(24):
    c = 2000
    data, y = get_data(rows = np.s_[(c*(i%6)):(c*((i%6)+1))], classes = ('0', '1'))
    model.advance(data, y)
    model.log_fitness()
best = model.best()

print(accuracy(model = best))
confusion_matrix(model = best)
plt.plot(model.log_max_fitness,label='pскр. = 0.75, pмут. = 0.05')
plt.plot(model.log_min_fitness,label='Мин. приспособленность', linewidth='8')
plt.plot(model.log_avg_fitness,label='Средняя приспособленность', linewidth='8')
plt.legend()
plt.show()


# model = ga(selection = sel_tournament, crossover = cross_swap_random_05, chromosome = chromosome_nn, population_size = 50, pool_size = 15, crossover_p = 1, mutation_p = 0, first_gen = 1)
# for i in range(24):
#     c = 2000
#     data, y = get_data(rows = np.s_[(c*(i%6)):(c*((i%6)+1))], classes = ('0', '1'))
#     model.advance(data, y)
#     model.log_fitness()
# plt.plot(model.log_max_fitness,label='pскр. = 1, pмут. = 0')
# model = ga(selection = sel_tournament, crossover = cross_swap_random_05, chromosome = chromosome_nn, population_size = 50, pool_size = 15, crossover_p = 0, mutation_p = 1, first_gen = 1)
# for i in range(24):
#     c = 2000
#     data, y = get_data(rows = np.s_[(c*(i%6)):(c*((i%6)+1))], classes = ('0', '1'))
#     model.advance(data, y)
#     model.log_fitness()
# plt.plot(model.log_max_fitness,label='pскр. = 0, pмут. = 1')
# model = ga(selection = sel_tournament, crossover = cross_swap_random_05, chromosome = chromosome_nn, population_size = 50, pool_size = 15, crossover_p = 1, mutation_p = 1, first_gen = 1)
# for i in range(24):
#     c = 2000
#     data, y = get_data(rows = np.s_[(c*(i%6)):(c*((i%6)+1))], classes = ('0', '1'))
#     model.advance(data, y)
#     model.log_fitness()
# plt.plot(model.log_max_fitness,label='pскр. = 1, pмут. = 1')
# model = ga(selection = sel_tournament, crossover = cross_swap_random_05, chromosome = chromosome_nn, population_size = 50, pool_size = 15, crossover_p = 0.75, mutation_p = 0.05, first_gen = 1)
# for i in range(24):
#     c = 12000
#     data, y = get_data(rows = np.s_[(c*(i%1)):(c*((i%1)+1))], classes = ('0', '1'))
#     model.advance(data, y)
#     model.log_fitness()
# plt.plot(model.log_max_fitness,label='pскр. = 0.75, pмут. = 0.05, nэлем. = 12000')
# model = ga(selection = sel_tournament, crossover = cross_swap_random_05, chromosome = chromosome_nn, population_size = 50, pool_size = 15, crossover_p = 1, mutation_p = 1, first_gen = 1)
# for i in range(24):
#     c = 250
#     data, y = get_data(rows = np.s_[(c*(i%48)):(c*((i%48)+1))], classes = ('0', '1'))
#     model.advance(data, y)
#     model.log_fitness()
# plt.plot(model.log_max_fitness,label='pскр. = 0.75, pмут. = 0.05, nэлем. = 250')

# plt.legend()
# plt.show()