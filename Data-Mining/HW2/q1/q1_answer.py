import matplotlib.pyplot as plt
import pandas as pd


def file_read(file_name):
    file_data_lines = open(file_name, 'r').readlines()
    data_array = []
    for flower in range(0, len(file_data_lines)):
        data_array.append([])
        file_data_line = file_data_lines[flower].split(',')
        for feature in range(0, len(file_data_line)):
            if feature < 4:
                data_array[flower].append(float(file_data_line[feature]))
            else:
                data_array[flower].append(file_data_line[feature].rstrip("\n"))
    return data_array


def data_frame_gen(data_array, column_names):
    data_frame = pd.DataFrame(data_array,
                              index=list(flower for flower in
                                         range(1, len(data_array) + 1)),
                              columns=list(column_names))
    # print(type(data_frame))
    return data_frame


def sub_frame_gen(data_frame, data_frame_features):
    grouped_data_frame = data_frame.groupby(data_frame['irisClass'])
    data_frame_features.remove('irisClass')
    for feature in data_frame_features:
        print("\nThe mean of %s for different flowers" % (feature))
        print(grouped_data_frame[feature].mean())
        print("\nThe standard deviation of %s for different flowers" % (
            feature))
        print(grouped_data_frame[feature].std())
    return grouped_data_frame


def plot_graph(data_frame, data_frame_features):
    data_frame_features.remove('irisClass')
    grouped_data_frame = data_frame.groupby(data_frame['irisClass']).apply(
        lambda tdf: pd.Series(dict([[vv, tdf[vv].tolist()] for vv in tdf if
                                    vv not in ['irisClass']])))
    flower_list = set(data_frame['irisClass'])
    for feature in data_frame_features:
        data = []
        fig, box_plotter = plt.subplots(figsize=(10, 6))
        box_plotter.yaxis.grid(True, linestyle='-', which='major',
                               color='lightgrey',
                               alpha=0.5)
        # Hide these grid behind plot objects
        box_plotter.set_axisbelow(True)
        box_plotter.set_xlabel('Distribution of flowers')
        box_plotter.set_ylabel('Length of features')
        box_plotter.set_title('Comparison of %s across flowers' % (feature))
        for flower in flower_list:
            data.append(grouped_data_frame[feature][flower])
        plt.figtext(0.8, 0.9,
                    '1 --> Iris-Versicolor\n2--> Iris-Virginica\n3--> '
                                                                'Iris-Setosa',
                    backgroundcolor='white', color='black', weight='roman',
                    size='medium')
        box_plotter.boxplot(data)
        plt.show()


def question1(data_frame, data_frame_features):
    data_frame_features.remove('irisClass')
    print(
        "\nPrinting Avgerage and Standard Deviation for each feature in "
        "Iris Data set\n")
    for column in data_frame_features:
        print('')
        print("The average for %s in the iris data set is %f" % (
            column, data_frame[column].mean()))
        print("The standard deviation for %s in the iris data set is %f" % (
            column, data_frame[column].std()))


def start_question1(file_name):
    data_array = file_read(file_name)
    column_names = ["Sepal Length", "Sepal Width", "Petal Length",
                    "Petal Width", "irisClass"]
    data_frame = data_frame_gen(data_array, column_names)
    data_frame_features = list(data_frame.columns.values)
    question1(data_frame, data_frame_features[:])
    sub_frame_gen(data_frame, data_frame_features[:])
    plot_graph(data_frame, data_frame_features[:])


start_question1("iris.data.txt")
