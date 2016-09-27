from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt


def file_read(file_name, attributes_name):
    data_frame = pd.read_csv(file_name, names=attributes_name)
    return data_frame


def pearson_correlation(data_frame):
    return pd.DataFrame.corr(data_frame, method='pearson', min_periods=1)


def make_list(sorted_list):
    final_list = []
    for item, i in zip(sorted_list, range(0, len(sorted_list))):
        if i % 2 == 0:
            final_list.append(item)
    return final_list[:4]


def select_correlated_pairs(pearson_correlation_matrix):
    temp_list = []
    for index in pearson_correlation_matrix.index.values:
        for column in pearson_correlation_matrix.columns.values:
            if index != column:
                temp_list.append([index, column, abs(
                    pearson_correlation_matrix[index][column])])
    return make_list(
        sorted(temp_list, key=itemgetter(2), reverse=False)), make_list(
        sorted(temp_list, key=itemgetter(2), reverse=True))


def get_ab(i):
    a, b = 0, 0
    if i == 1:
        b = 1
    elif i == 2:
        a = 1
    elif i == 3:
        a, b = 1, 1
    return a, b


def scatter_plot(list, data_frame, title):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(title, fontsize=20)
    for item, i in zip(list, range(0, len(list))):
        a, b = get_ab(i)
        data_frame.plot(kind='scatter', x=item[0], y=item[1], ax=axes[a, b],
                        title="Ranked %d" % (i + 1) + "with correlation %f" % (
                        item[2]))
    fig.tight_layout()
    # plt.show()


def start_question2(file_name):
    attributes_name = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']
    data_frame = file_read(file_name, attributes_name[:])
    pearson_correlation_matrix = pearson_correlation(data_frame)
    # print(pearson_correlation_matrix)
    top4_list, last4_list = select_correlated_pairs(pearson_correlation_matrix)
    scatter_plot(last4_list, data_frame, "Least Correlated Pairs")
    scatter_plot(top4_list, data_frame, "Top Correlated Pairs")


start_question2("wine.data.txt")
plt.show()
