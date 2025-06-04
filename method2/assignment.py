import os
import argparse
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
import time
import random
import numpy as np

# data
user_num = 0
raw_data = []
user_data_num = []
max_item_num = 0

# sampling
train_data_p = set()
train_data_n = set()
test_data_p = set()
sample_ratio = 0.8
data_p = set()
data_n = set()
negative_sample_ratio = 1
data_n_sample = set()

# vector
hidden_factors = 16
model = None
user_vector = np.array([])
item_vector = np.array([])

# bias
user_bias = np.array([])
item_bias = np.array([])

def load():
    global user_num, raw_data, user_data_num, max_item_num
    print("Loading...")
    with open('train.csv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            user_data_num.append(0)
            parts = line.split(",")
            items = parts[1].split(" ")
            for i in items:
                raw_data.append((int(parts[0]), int(i))) 
                user_data_num[user_num] += 1
                max_item_num = max(max_item_num, int(i))
            user_num += 1
    print("Loading done.")

def sampling():
    print("Sampling...")
    global train_data, test_data
    prefix = 0
    for i in range(user_num):
        for j in range(user_data_num[i]):
            data_p.add(raw_data[prefix + j])
            if j < math.floor(user_data_num[i] * sample_ratio):
            # if np.random.rand() < sample_ratio:
                train_data_p.add(raw_data[prefix + j])
            else:
                test_data_p.add(raw_data[prefix + j])
        prefix += user_data_num[i]
        for j in range(math.floor(user_data_num[i] * sample_ratio * negative_sample_ratio)):
            while True:
                item = random.randint(0, max_item_num)
                if (i, item) not in train_data_p and (i, item) not in train_data_n:
                    train_data_n.add((i, item))
                    break
        for j in range(math.floor(user_data_num[i])):
            while True:
                item = random.randint(0, max_item_num)
                if (i, item) not in data_p:
                    data_n.add((i, item))
                    break
        for j in range(math.floor(user_data_num[i] * negative_sample_ratio)):
            while True:
                item = random.randint(0, max_item_num)
                if (i, item) not in data_p and (i, item) not in data_n_sample:
                    data_n_sample.add((i, item))
                    break
    print("Sampling done.")

def init_vector():
    global hidden_factors, user_num, max_item_num, user_vector, item_vector, user_bias, item_bias
    print("Initializing vector...")
    # user_vector = [[random.random() for _ in range(hidden_factors)] for _ in range(user_num)]
    # item_vector = [[random.random() for _ in range(hidden_factors)] for _ in range(max_item_num)]
    user_vector = np.random.uniform(low=0.0, high=1.0, size=(user_num, hidden_factors))
    # item_vector = np.random.uniform(low=0.0, high=1.0, size=(max_item_num+1, hidden_factors))
    # user_vector = np.zeros((user_num, hidden_factors))
    item_vector = np.zeros((max_item_num+1, hidden_factors))
    # 初始化bias項
    # user_bias = np.random.uniform(low=0.0, high=1.0, size=user_num)
    # item_bias = np.random.uniform(low=0.0, high=1.0, size=max_item_num+1)
    print("Vector initialization done.")

def sigma(x):
    return 1 / (1 + math.exp(-x))

def BCE_gradient(rate = 0.01, iterations = 30, data_p = train_data_p, data_n = train_data_n):
    print("BCE gradient...")
    global user_vector, item_vector
    data_p_labeled = np.array([[*sample, 1] for sample in data_p])
    data_n_labeled = np.array([[*sample, -1] for sample in data_n])
    train = np.concatenate((data_p_labeled, data_n_labeled))
    np.random.shuffle(train)
    for _ in range(iterations):
        print(f"Iteration {_+1}/{iterations}: {calculate_MAP()}")
        # print(BCE_evluate(data_p = test_data_p))
        for sample in train:
            # print(sample)
            user = int(sample[0])
            item = int(sample[1])
            label = sample[2]
            user_vec = user_vector[user]
            item_vec = item_vector[item]
            # 加入bias項的計算
            # interaction = np.dot(user_vec, item_vec) + user_bias[user] + item_bias[item]
            # loss_grad = label*sigma(-1*label*interaction)
            loss_grad = label*sigma(-1*label*np.dot(user_vec, item_vec))
            # user_vector[user] += rate*np.dot(loss_grad, item_vec)
            # item_vector[item] += rate*np.dot(loss_grad, user_vec)
            user_vector[user] += rate*loss_grad*item_vec
            item_vector[item] += rate*loss_grad*user_vec
            # user_bias[user] += rate*loss_grad
            # item_bias[item] += rate*loss_grad
    print("BCE gradient done.")

def BCE_gradient_alln(rate = 0.01, iterations = 30, data_p = train_data_p):
    print("BCE gradient...")
    global user_vector, item_vector
    for _ in range(iterations):
        print(f"Iteration {_+1}/{iterations}: {calculate_MAP()}")
        # print(BCE_evluate(data_p = test_data_p))
        for j in range(max_item_num*user_num):
            user = np.random.randint(0, user_num-1)
            item = np.random.randint(0, max_item_num)
            label = 1 if (user, item) in data_p else -1
            user_vec = user_vector[user]
            item_vec = item_vector[item]
            loss_grad = label*sigma(-1*label*np.dot(user_vec, item_vec))
            user_vector[user] += rate*loss_grad*item_vec
            item_vector[item] += rate*loss_grad*user_vec
    print("BCE gradient done.")

def BCE_evluate(data_p):
    # print("BCE evaluate...")
    BCEloss = 0
    for sample in data_p:
        user = int(sample[0])
        item = int(sample[1])
        BCEloss += math.log(sigma(np.dot(user_vector[user], item_vector[item])))
    for i in range(len(data_p)):
        while True:
            user = random.randint(0, user_num - 1)
            item = random.randint(0, max_item_num)
            if (user, item) not in train_data_p and (user, item) not in test_data_p:
                # interaction = np.dot(user_vector[user], item_vector[item]) + user_bias[user] + item_bias[item]
                # BCEloss += math.log(1 - sigma(interaction))
                BCEloss += math.log(1 - sigma(np.dot(user_vector[user], item_vector[item])))
                break
    return -BCEloss

def BPR_gradient(epoch = 0.96, rate = 0.1, iterations = 100, lam = 0.009,  data_p = data_p, data_n = train_data_n):
    print("BPR gradient...")
    global user_vector, item_vector
    for _ in range(iterations*100000):
        if _ % 100000 == 0:
            rate *= epoch
            print(f"Iteration {_/100000 + 1}/{iterations}: {calculate_MAP()}")
            # print(calculate_MAP())
            # print(BPR_evluate(data_p = test_data_p, data_n = train_data_n, lam = lam))
        user = np.random.randint(0, user_num-1)
        while True:
            item_p = np.random.randint(0, max_item_num)
            if (user, item_p) in data_p:
                break
        while True:
            item_n = np.random.randint(0, max_item_num)
            if (user, item_n) not in data_p:
                break
        user_vec = user_vector[user]
        item_vec_p = item_vector[item_p]
        item_vec_n = item_vector[item_n]
        tmp = 1-sigma(np.dot(user_vec, item_vec_p) - np.dot(user_vec, item_vec_n))
        user_vector[user] += rate * (np.dot(tmp, (item_vec_p - item_vec_n)) - 2* lam * user_vec)
        item_vector[item_p] += rate * (tmp * user_vec - 2 * lam * item_vec_p)
        item_vector[item_n] += rate * (-tmp * user_vec - 2 * lam * item_vec_n)
        # for p in data_p:
        #     for n in data_n:
        #         if(p[0] == n[0]):
        #             user = int(p[0])
        #             item_p = int(p[1])
        #             item_n = int(n[1])
        #             user_vec = user_vector[user]
        #             item_vec_p = item_vector[item_p]
        #             item_vec_n = item_vector[item_n]
        #             tmp = 1-sigma(np.dot(user_vec, item_vec_p) - np.dot(user_vec, item_vec_n))
        #             user_vector[user] += rate * (np.dot(tmp, (item_vec_p - item_vec_n)) - 2* lam * user_vec)
        #             item_vector[item_p] += rate * (tmp * user_vec - 2 * lam * item_vec_p)
        #             item_vector[item_n] += rate * (-tmp * user_vec - 2 * lam * item_vec_n)
        

def BPR_evluate(data_p, data_n, lam):
    # print("BPR evaluate...")
    global user_vector, item_vector
    BPRloss = 0
    used_user = set()
    used_item = set()
    for _ in range(max_item_num):
        user = np.random.randint(0, user_num-1)
        while True:
            item_p = np.random.randint(0, max_item_num)
            if (user, item_p) in data_p:
                break
        while True:
            item_n = np.random.randint(0, max_item_num)
            if (user, item_n) not in data_p:
                break
        user_vec = user_vector[user]
        item_vec_p = item_vector[item_p]
        item_vec_n = item_vector[item_n]
        BPRloss += math.log(sigma(np.dot(user_vec, item_vec_p) - np.dot(user_vec, item_vec_n))) 
        if user not in used_user:
            used_user.add(user)
            BPRloss += lam * (np.dot(user_vec, user_vec))
        if item_p not in used_item:
            used_item.add(item_p)
            BPRloss += lam * (np.dot(item_vec_p, item_vec_p))
        if item_n not in used_item:
            used_item.add(item_n)
            BPRloss += lam * (np.dot(item_vec_n, item_vec_n))
    # for p in data_p:
    #     for n in data_n:
    #         if(p[0] == n[0]):
    #             user = int(p[0])
    #             item_p = int(p[1])
    #             item_n = int(n[1])
    #             user_vec = user_vector[user]
    #             item_vec_p = item_vector[item_p]
    #             item_vec_n = item_vector[item_n]
    #             BPRloss += math.log(sigma(np.dot(user_vec, item_vec_p) - np.dot(user_vec, item_vec_n))) 
    #             if user not in used_user:
    #                 used_user.add(user)
    #                 BPRloss += lam * (np.dot(user_vec, user_vec))
    #             if item_p not in used_item:
    #                 used_item.add(item_p)
    #                 BPRloss += lam * (np.dot(item_vec_p, item_vec_p))
    #             if item_n not in used_item:
    #                 used_item.add(item_n)
    #                 BPRloss += lam * (np.dot(item_vec_n, item_vec_n))
    return -BPRloss

def calculate_MAP(k=50):
    AP_sum = 0
    user_count = 0
    for user in range(user_num):
        true_items = set([item for (u, item) in test_data_p if u == user])
        if not true_items:
            continue

        scores = []
        for item in range(max_item_num + 1):
            if (user, item) not in train_data_p:
                score = np.dot(user_vector[user], item_vector[item])
                scores.append((item, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [item for item, _ in scores[:k]]
        
        hit = 0
        AP = 0
        for rank, item in enumerate(recommended, start=1):
            if item in true_items:
                hit += 1
                AP += hit / rank
        AP /= len(true_items)
        AP_sum += AP
        user_count += 1
    return round(AP_sum / user_count, 4) if user_count > 0 else 0

def save_vectors():
    global user_vector, item_vector, user_num, max_item_num
    print("Saving vectors...")
    np.save('user_vector.npy', user_vector)
    np.save('item_vector.npy', item_vector)
    np.save('train_data_p.npy', np.array(list(train_data_p)))
    np.save('test_data_p.npy', np.array(list(test_data_p)))
    
    # Also save metadata needed for the second program
    with open('metadata.txt', 'w') as f:
        f.write(f"{user_num}\n")
        f.write(f"{max_item_num}\n")
    print("Vectors saved.")

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='IR HW5')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-d', '--hidden_factors', required=True)
    args = parser.parse_args()
    model = args.model
    hidden_factors = int(args.hidden_factors)
    np.random.seed(817)
    load()
    sampling()
    init_vector()
    if model == "BCE":
        # BCE_gradient(data_p = train_data_p, data_n = train_data_n)
        # print(BCE_evluate(data_p = test_data_p))
        # BCE_gradient(data_p = data_p, data_n = data_n_sample)
        # BCE_gradient_alln(data_p = train_data_p)
        # print(BCE_evluate(data_p = test_data_p))
        BCE_gradient_alln(data_p = data_p)
    elif model == "BPR":
        # BPR_gradient(data_p = train_data_p, data_n = train_data_n)
        BPR_gradient(data_p = data_p, data_n = data_n)
    save_vectors()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} sec")