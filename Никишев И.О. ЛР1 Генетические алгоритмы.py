#ФИО Никишев Иван Олегович nkshv2@gmail.com, учебный курс https://online.mospolytech.ru/mod/assign/view.php?id=281548, год 2023

import random, numpy as np, math
from functools import partial
seed = 3
random.seed(seed)
np.random.seed(seed)

def dist(connections, distances, point0, point1, i):
    try:return distances[connections.index([point0, point1])]
    except ValueError:
        try:return distances[connections.index([point1, point0])]
        except ValueError: return 9999999999/i

def exists(connections, point0, point1):
    if [point0, point1] in connections or [point1, point0] in connections: return True
    else: return False

def distance(connections, distances, path, end):
    distance = 0
    for i in range(1, len(path)):
        distance += dist(connections, distances, path[i-1], path[i], i)
        if path[i] == end: return distance
    return 0
    
def random_network(n_nodes, n_connections, xmax, ymax, max_distance):
    nodes=[]
    connections = []
    distances = []
    x = random.uniform(0, xmax)
    y =  random.uniform(0, ymax)
    for i in range(n_nodes):
        while [x, y] in nodes:
            x = random.uniform(0, xmax)
            y =  random.uniform(0, ymax)
        nodes.append([x, y])
    nodes = sorted(nodes, key = lambda i: i[0])
    for i in range(n_connections):
        c =[random.randrange(0, n_nodes),random.randrange(0, n_nodes), ]
        limit=0
        stop = False
        while c in connections or c[0] == c[1] or math.dist(nodes[c[0]], nodes[c[1]]) > max_distance:
            c =[random.randrange(0, n_nodes),random.randrange(0, n_nodes), ]
            limit+=1
            if limit == 10000: 
                stop = True
                break
        if stop is True: break
        distances.append(math.dist(nodes[c[0]], nodes[c[1]]))
        connections.append(c)
    start, end = [random.randrange(0, n_nodes),random.randrange(0, n_nodes), ]
    while start==end or math.dist( nodes[start], nodes[end]) < xmax/2:
        print(math.dist( nodes[start], nodes[end]) , xmax/2)
        start, end = [random.randrange(0, n_nodes),random.randrange(0, n_nodes), ]
    return nodes, connections, distances, start, end

def squares_network(x, y, connection_chance):
    nodes=[[i//x, i%x] for i in range(x*y)]
    connections = []
    distances = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if math.dist(nodes[i], nodes[j]) == 1 and random.uniform(0,1) <= connection_chance:
                connections.append([i, j])
                distances.append(1)
    return nodes, connections, distances

def sel_roulette(population, fitness, num):
    """приспособленность особи является вероятностью её выбора"""
    pool=[]
    fitness = np.array([i if i>0 else 0 for i in fitness])
    if np.count_nonzero(fitness!= 0) <= num:
        return sel_tournament(population, fitness, num)
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

def cross_pmx(i1, i2, cut_size=None):
    """Берет одинаковый случайный разрез заданного размера из обеих особей. 
    Каждое число из разреза одной особи переходит на ту же позицию второй
    особи, меняясь местами с изначальной позицией если она присутствует"""
    start = i1[0]
    end = i1 [-1]
    res= [i1.copy()[1:-1], i2.copy()[1:-1]]
    i12= [i1.copy()[1:-1], i2.copy()[1:-1]]
    if cut_size is None: cut_size = random.randrange(1, len(i1)-2)
    cut_s = random.randint(0, len(i1) - cut_size - 2)
    for j in range(2):
        #print(res[j], f'j = {j}, cut_s = {cut_s}, cut_size = {cut_size}')
        for i in range(cut_size):
            try: 
                pos = res[j].index(i12[j-1][cut_s+i]) # индекс числа 2го у 1го
                res[j][cut_s+i], res[j][pos] = res[j][pos], res[j][cut_s+i] # число 2 переходит на ту же позицию 1, меняясь местами с изначальной позицией
            except ValueError:
                res[j][cut_s+i] = i12[j-1][cut_s+i]
    res[0].insert(0, start)
    res[0].append(end)
    res[1].insert(0, start)
    res[1].append(end)
    return res[0], res[1]

def cross_cut(i1, i2):
    cut_pos = random.randrange(2,len(i1)-2)
    res1 = i1.copy()
    res2 = i2.copy()
    res1[cut_pos:] = i2[cut_pos:]
    res2[cut_pos:] = i1[cut_pos:]
    return res1, res2

def mutate_swap(ind):
    """2 числа меняются местами"""
    a=0
    b=0
    while a==b: 
        a=random.randrange(1, len(ind)-1)
        b=random.randrange(1, len(ind)-1)
    ind[a], ind[b] = ind[b], ind[a]
    return ind

def mutate_replace(ind):
    """2 числа меняются местами"""
    if random.uniform(0, 1)>0.5 or ind[1]: position=random.randrange(1, len(ind)-1)
    else: position = random.randint(1, ind.index(ind[-1]))
    value=random.randrange(1, len(ind)-1)
    while position==value or value == ind[0] or value == ind[-1]: 
        position=random.randrange(1, len(ind)-1)
        value=random.randrange(1, len(ind)-1)
    if value in ind: ind[position], ind[ind.index(value)] = ind[ind.index(value)], ind[position]
    else: ind[position] = value
    return ind

def visualize(nodes, connections, path, exitonclick = True):
    import turtle
    nodesz = list(zip(*nodes))
    turtle.hideturtle()
    turtle.setworldcoordinates(min(nodesz[0])-1, min(nodesz[1])-1, max(nodesz[0])+1, max(nodesz[1])+1)
    turtle.tracer(False)
    turtle.clear()
    for i in nodes:
        turtle.penup()
        turtle.goto(i)
        turtle.pendown()
        turtle.dot()
    turtle.pencolor('yellow')
    for i in connections:
        turtle.penup()
        turtle.goto(nodes[i[0]])
        turtle.pendown()
        turtle.goto(nodes[i[1]])
    turtle.penup()
    turtle.goto(nodes[path[0]][0], nodes[path[0]][1])
    turtle.pendown()
    #print(path)
    for c, v in enumerate(path):
        if exists(connections, path[c], path[c-1]) or c==0:
            turtle.pencolor('black')
        else:turtle.pencolor('red')
        turtle.goto(*nodes[v])
        if v == end: break
    turtle.penup()
    turtle.goto(nodes[start])
    turtle.pendown()
    turtle.pencolor('blue')
    turtle.dot()
    turtle.penup()
    turtle.goto(nodes[end])
    turtle.pendown()
    turtle.pencolor('green')
    turtle.dot()
    turtle.tracer(True)
    if exitonclick: turtle.exitonclick()

def statistics(log):
    log = np.array(log)
    import matplotlib.pyplot as plt
    plt.plot(log[:,0], label ='max')
    plt.plot(log[:,1], label ='min')
    plt.plot(log[:,2], label ='avg')
    plt.title('ФИО Никишев Иван Олегович; nkshv2@gmail.com; Учебный курс: https://online.mospolytech.ru/mod/assign/view.php?id=281548; год 2023') 
    plt.ylabel('Длина пути')
    plt.xlabel('Поколение\nФИО Никишев Иван Олегович; nkshv2@gmail.com; Учебный курс: https://online.mospolytech.ru/mod/assign/view.php?id=281548; год 2023')
    plt.legend(loc='best')
    plt.show()

def chromosome_path(nodes, connections, start, end):
    nodes=nodes.copy()
    chromosome = [start]
    del nodes[start]
    end_position = random.randrange(1, len(nodes)-1)
    indexes = list(range(len(nodes)))
    del indexes[end-1]
    for i in range(1, len(nodes)-1):
        if exists(connections, chromosome[i-1], end):
            chromosome.append(end)
        else: 
            possible = []
            for j, value in enumerate(connections):
                if chromosome[i-1] in value:
                    if value[0] == chromosome[i-1]: v = value[1]
                    else: v  = value[0]
                    #if v not in chromosome: possible.append(v)
                    possible.append(v)
            if len(possible)>0: chromosome.append(random.choice(possible))
            else: chromosome.append(random.choice([j for j in indexes if j not in chromosome]))
    if end not in chromosome: chromosome[end_position] = end
    chromosome[-1] = end
    return chromosome 

def chromosome_path_simple(nodes, connections, start, end):
    nodes=nodes.copy()
    chromosome = [start]
    del nodes[start]
    end_position = random.randrange(1, len(nodes)-1)
    indexes = list(range(len(nodes)))
    del indexes[end-1]
    chromosome.extend(random.sample(indexes, len(nodes)-2))
    if end not in chromosome: chromosome[end_position] = end
    chromosome[-1] = end
    return chromosome


def genetic(problem, selection, crossover, mutation, chromosome, population_size = 200, generations = 1000, pool_size = 25, crossover_p = 0.7, mutation_p = 0.01, first_gen = 5, mode='пошаговый', **kwargs):
    population=[chromosome() for i in range(int(population_size*first_gen))]
    #for i in population:visualize(nodes=kwargs['nodes'], connections=kwargs['connections'], path = i, exitonclick=False)
    log=[]
    for gen in range(1, generations):
        # СЕЛЕКЦИЯ
        population_prev = population.copy()
        fitness = [problem(path=i) for i in population]
        #print(min(fitness))
        pool = selection(population.copy(), [max(fitness)-i for i in fitness], num =pool_size)
        #for i in pool: visualize(nodes=kwargs['nodes'], connections=kwargs['connections'], path = i, exitonclick=False)
        population=[population[fitness.index(min(fitness))]]
        i=1
        n_crossovers = 0
        if mode == 'пошаговый': print(f'{gen} поколение: {population_prev};\nРодительсьская популяция: {pool}')
        while i < population_size:
            if random.uniform(0,1)<crossover_p:
                # СКРЕЩИВАНИЕ
                selected_inds = random.sample(pool, 2)
                crossover_inds = crossover(*selected_inds)
                while any([i in population for i in crossover_inds]):
                    n_crossovers+=1
                    selected_inds = random.sample(pool, 2)
                    crossover_inds = crossover(*selected_inds)
                population.extend(crossover_inds)
                i+=len(crossover_inds)
                if mode == 'пошаговый': print(f'Особи {i-len(crossover_inds)} : {i} получены скрещиванием 2-х особей:\n{selected_inds[0]}\n{selected_inds[1]}\nРезультат:\n{crossover_inds[0]}\n{crossover_inds[1]}')
            else:
                ind = random.choice(pool)
                n_rand = 0
                while ind in population: 
                    ind = random.choice(pool)
                    n_rand+=1
                    if n_rand>=100: 
                        if random.uniform(0,1)>0.5:
                            ind = random.choice(population_prev)
                        else:  ind = mutation(ind.copy())
                population.append(ind)
                i+=1
        #print(n_crossovers, n_rand)
        for i in range(1, len(population)):
            if random.uniform(0,1)<mutation_p:
                # МУТАЦИЯ
                if mode == 'пошаговый': print(f'Особь {i} до и после применения оператора мутации:\n{population[i]}')
                ind = mutation(population[i].copy())
                while ind in population: ind = mutation(ind)
                population[i] = ind
                if mode == 'пошаговый': print(f'{population[i]}')
        
        fitness_good = [i for i in fitness if i < 999999]
        if len(fitness_good)!=0: fitness = fitness_good
        max_fitness = max(fitness)
        min_fitness = min(fitness)
        avg_fitness = sum(fitness)/len(fitness)
        log.append([min_fitness, max_fitness, avg_fitness])
        print(f'Поколение: {gen}, мин. приспособленность: {min_fitness}, макс: {max_fitness}, средняя: {avg_fitness}')
        if mode == 'пошаговый': visualize(nodes=kwargs['nodes'], connections=kwargs['connections'], path = population[0], exitonclick=False)
        if mode == 'пошаговый': input(f'Сформировано поколение {gen}.\n')
    return population[fitness.index(min(fitness))], log


n_nodes = 75
nodes, connections, distances, start, end = random_network(n_nodes, n_connections = 500, xmax = 100, ymax = 100, max_distance=25)
#nodes, connections, distances, start, end = squares_network(10, 10, 1), 0, 90
start, end = 0, n_nodes-1
problem = partial(distance, connections = connections, distances = distances, end = end)
chromosome = partial(chromosome_path, nodes, connections, start, end)
solution, log = genetic(problem=problem, selection=sel_roulette, crossover = cross_cut, mutation=mutate_replace, chromosome=chromosome, generations=2000, population_size = 100, pool_size=50, crossover_p=0.75, mutation_p=1, nodes=nodes, connections=connections, first_gen = 1, mode = 'пошаговы')
print(start, end)
visualize(nodes, connections, solution)

statistics(log)