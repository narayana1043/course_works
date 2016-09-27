from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from filecmp import cmp


def file_read(file_name, attributes_name):
    data_frame = pd.read_csv(file_name, names=attributes_name)
    return data_frame


def pearson_correlation(data_frame):
    return (pd.DataFrame.corr(data_frame, method='pearson', min_periods=1))


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
        sorted(temp_list, key=itemgetter(2), reverse=True)), make_list(
        sorted(temp_list, key=itemgetter(2)))


def eu_distance(data_frame):
    class_count = [0, 0, 0]
    eu_class_match_count = [0, 0, 0]
    tot_matched = 0
    tot_count = 0
    for item in data_frame.iterrows():
        eu_list = []
        # print(item[1])
        for item_next in data_frame.iterrows():
            if item[0] != item_next[0]:
                eu_list.append(
                    [distance.euclidean(item[1], item_next[1]), item_next])
        # print(eu_list)
        eu_close_match = sorted(eu_list, key=lambda x: x[0])[0]
        # print("Best match for \n %s \n is \n %s"%(item,eu_close_match))
        if item[1][0] == 1:
            class_count[0] += 1
            if item[1][0] == eu_close_match[1][1]['Class']:
                eu_class_match_count[0] += 1
        elif item[1][0] == 2:
            class_count[1] += 1
            if item[1][0] == eu_close_match[1][1]['Class']:
                eu_class_match_count[1] += 1
        elif item[1][0] == 3:
            class_count[2] += 1
            if item[1][0] == eu_close_match[1][1]['Class']:
                eu_class_match_count[2] += 1
    for i in range(3):
        tot_matched += eu_class_match_count[i]
        tot_count += class_count[i]
        print("Percentage of points in class%d whose closest neighbors have "
              "class%d is %f" % (i + 1, i + 1, (eu_class_match_count[i] /
                                                class_count[i]) * 100))
    # print("Percentage of points in class2 whose closest neighbors have "
    #       "class1 is %d" % (eu_class_match_count[1] / len(data_frame) * 100))
    # print("Percentage of points in class3 whose closest neighbors have "
    #       "class1 is %d" % (eu_class_match_count[2] / len(data_frame) * 100))
    print("Percentage of points in the whole dataset whose closest neighbors "
          "have same class is %f" % ((tot_matched / tot_count) * 100))


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
                        title="Ranked %d" % (
                            i + 1) + " with correlation %f" % (item[2]))
    fig.tight_layout()


def start_question2(file_name):
    attributes_name = ['Class', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']
    data_frame = file_read(file_name, attributes_name[:])
    # print(data_frame.head(3))
    col_to_norm = list(data_frame.columns)
    col_to_norm.remove('Class')
    data_frame[col_to_norm] = data_frame[col_to_norm].apply(
        lambda x: (x - x.mean()) / (x.max() - x.min()))
    # print(data_frame.head(3))
    pearson_correlation_matrix = pearson_correlation(data_frame[col_to_norm])
    top4_list, last4_list = select_correlated_pairs(pearson_correlation_matrix)
    scatter_plot(last4_list, data_frame, "Least Correlated Pairs")
    scatter_plot(top4_list, data_frame, "Top Correlated Pairs")
    eu_distance(data_frame)


start_question2("wine.data.txt")
plt.show()
