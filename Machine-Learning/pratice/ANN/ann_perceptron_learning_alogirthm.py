import numpy as np
import pandas as pd
import random
from math import sqrt

def compute_predicted_output(weighted_vector, each_training_example,len_df_cols, theta, bias):
    output = sum(weighted_vector[i]*each_training_example[1][i] for i in range(len_df_cols)) + bias
    return output


def ann():
    df = pd.read_csv(filepath_or_buffer='./dataset', sep=',', header=0, dtype=np.float64)
    len_df_cols = len(df.columns.values)
    weighted_vector = [random.random() for i in range(len_df_cols)]
    bias = random.random()
    # learning rate: if learning rate is close to '0' then the new weight is mostly influenced by the old weight
    # if learning rate is close to '1' then the new weight is sensitive to the amount of adjustment performed in the current iteration
    learning_rate = 0.5
    global_error = 0
    theta = 0

    for each_training_example in df.iterrows():
        activation_output = compute_predicted_output(weighted_vector, each_training_example, len_df_cols, theta, bias)
        local_error = activation_output - each_training_example[1]['y']
        for each_weight in range(len(weighted_vector)):
            weighted_vector[each_weight] += learning_rate * local_error * each_training_example[1][each_weight]
        bias += (learning_rate * local_error)

    print('RMS Error: ', sqrt(global_error))
    print(weighted_vector[0], 'x1 + ', weighted_vector[1], 'x2 + ', weighted_vector[2], 'x3 + ', bias, ' = 0')

    count = 0
    for each_training_example in df.iterrows():
        output = compute_predicted_output(weighted_vector, each_training_example, len_df_cols, theta, bias)
        print(each_training_example[1][3])
        if output >= 1:
            count += 1

    accuracy = (count/df.shape[0]) * 100
    print(accuracy)

ann()

