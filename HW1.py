import numpy as np
import time
from random import randint


""" load dataset from file """
def loadDataset(filename):
    dataset = np.array([[[0, 0, 0, 0, 0], 0]])
    with open(filename) as f:
        raw_data = next(f).split()
        X = np.array([1] + [float(x) for x in raw_data[:-1]])
        y = int(raw_data[-1])
        dataset = np.array([[X, y]])

        for line in f:
            raw_data = line.split()
            X = np.array([1] + [float(x) for x in raw_data[:-1]])
            y = int(raw_data[-1])
            dataset = np.append(dataset, [[X, y]], axis=0)

    return dataset


""" PLA alg. by naive cycle """
def PLA(dataset, eta=1):
    length = len(dataset)
    w = np.zeros(5)
    correct_data_streak = 0
    index = 0
    finished = False
    update_count = 0

    while not finished:
        X, y = dataset[index]
        sign = 1 if w.dot(X) > 0 else -1
        if (sign == y):
            correct_data_streak += 1
        else:
            w += eta * y * X
            update_count += 1
            correct_data_streak = 0

        index = (index + 1) % length
        if (correct_data_streak == length):
            finished = True
        
    return w, update_count


""" get error rate of dataset with w """
def get_error_rate(dataset, w):
    error_count = 0
    for X, y in dataset:
        sign = 1 if w.dot(X) > 0 else -1
        if (sign != y):
            error_count += 1
    return error_count / len(dataset)


""" PLA alg. w/ pocket alg. """
def PLA_pocket(dataset, update=50):
    length = len(dataset)
    w = np.zeros(5)
    best_w = np.zeros(5)
    best_error_rate = get_error_rate(dataset, best_w)
    update_count = 0

    while(update_count < update):
        X, y = dataset[randint(0, length - 1)]
        sign = 1 if w.dot(X) > 0 else -1
        if (sign != y):
            w += y * X
            update_count += 1
            error_rate = get_error_rate(dataset, w)

            if (error_rate < best_error_rate):
                best_w = np.copy(w)
                best_error_rate = error_rate

    return best_w, w


# Q.15

dataset = loadDataset("hw1_15_train.txt")
w, update_count = PLA(dataset)
print("Q.15 update count: %s" % update_count)
# Ans.15: update count = 45


# Q.16

shuffle_loop = 2000
avg_update_count = 0
for _ in range(shuffle_loop):
    np.random.seed(int(time.time()))
    np.random.shuffle(dataset)
    w, update_count = PLA(dataset)
    avg_update_count += update_count / shuffle_loop
print("Q.16 avg. update count: %s" % avg_update_count)
# Ans.16: avg. update count ≈ 40.016


# Q.17

avg_update_count = 0
for _ in range(shuffle_loop):
    np.random.seed(int(time.time()))
    np.random.shuffle(dataset)
    w, update_count = PLA(dataset, eta=0.5)
    avg_update_count += update_count / shuffle_loop
print("Q.17 avg. update count: %s" % avg_update_count)
# Ans.17: avg. update count ≈ 39.998


# Q.18 and Q.19

training_set = loadDataset("hw1_18_train.txt")
test_set = loadDataset("hw1_18_test.txt")
Q18_avg_error_rate = 0
Q19_avg_error_rate = 0
for i in range(shuffle_loop):
    np.random.seed(int(time.time()))
    np.random.shuffle(training_set)
    best_w, w = PLA_pocket(training_set, update=50)
    Q18_error_rate = get_error_rate(test_set, best_w)
    Q18_avg_error_rate += Q18_error_rate / shuffle_loop
    Q19_error_rate = get_error_rate(test_set, w)
    Q19_avg_error_rate += Q19_error_rate / shuffle_loop
print("Q.18 avg. error rate: %s" % Q18_avg_error_rate)
print("Q.19 avg. error rate: %s" % Q19_avg_error_rate)
# Ans.18: avg. error rate ≈ 0.13414
# Ans.19: avg. error rate ≈ 0.35428


# Q.20

avg_error_rate = 0
for i in range(shuffle_loop):
    np.random.seed(int(time.time()))
    np.random.shuffle(training_set)
    w, _ = PLA_pocket(training_set, update=100)
    error_rate = get_error_rate(test_set, w)
    avg_error_rate += error_rate / shuffle_loop
print("Q.20 avg. error rate: %s" % avg_error_rate)
# Ans.20: avg. error rate ≈ 0.11669
