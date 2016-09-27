import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import random
from collections import Counter


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    # knn Algo
    distances = []

    for group in data:
        for features in data[group]:
            eculidean_distance = np.linalg.norm(
                np.array(features) - np.array(predict))
            distances.append([eculidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    # print(vote_result, confidence)

    return vote_result, confidence


accuracies = []

# it will take longer time to run compared to sklearn is because it can be
#  threaded heavily and sklearn uses threading with n_jobs parameter

for i in range(25):
    df = pd.read_csv(
        './uci_breast_cancer_dataset/breast-cancer-wisconsin.data.txt')
    # print(df.head())
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # astype(float) to convert all the data to folat variables
    # .values.tolist() is not to loose the relationship of the features
    #  when shuffled
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)
    # print(20*'#')
    # print(full_data[:5])

    test_size = 0.3
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            # else:
                # print(confidence)
            total += 1

    # print('Accuracy:', correct / total)
    accuracies.append(correct / total)

print(sum(accuracies) / len(accuracies))
