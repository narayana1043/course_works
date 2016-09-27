import matplotlib as mpl
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
        print("\nThe mean of %s for different folwers" % (feature))
        print(grouped_data_frame[feature].mean())
    return grouped_data_frame


def plot_graph(data_frame, data_frame_features):
    data_frame_features.remove('irisClass')
    grouped_data_frame = data_frame.groupby(data_frame['irisClass']).apply(
        lambda tdf: pd.Series(dict([[vv, tdf[vv].tolist()] for vv in tdf if
                                    vv not in ['irisClass']])))
    # print(grouped_data_frame)
    flower_list = set(data_frame['irisClass'])
    print(flower_list)
    for feature in data_frame_features:
        data = []
        mpl.use('agg')
        fig, box_plotter = plt.subplots(figsize=(10, 6))
        box_plotter.yaxis.grid(True, linestyle='-', which='major',
                               color='lightgrey', alpha=0.5)
        # Hide these grid behind plot objects
        box_plotter.set_axisbelow(True)
        box_plotter.set_xlabel('Distribution of flowers')
        box_plotter.set_ylabel('Value')
        box_plotter.set_title('Comparison of %s across flowers' % (feature))
        # box_plotter.set_xticklabels(flower_list)
        for flower in flower_list:
            data.append(grouped_data_frame[feature][flower])
        box_plotter.set_xticklabels(['a', 'b', 'c'])
        box_plotter.boxplot(data, patch_artist=True)
        plt.show()


def question1(data_frame, data_frame_features):
    data_frame_features.remove('irisClass')
    print("\nPrinting Avgerage and Standard Deviation for each feature in Iris"
          " Data set\n")
    for column in data_frame_features:
        print("The average for %s in the iris data set is %f"
              % (column, data_frame[column].mean()))


def start_question1(file_name):
    data_array = file_read(file_name)
    column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth",
                    "irisClass"]
    data_frame = data_frame_gen(data_array, column_names)
    data_frame_features = list(data_frame.columns.values)
    # question1(data_frame, data_frame_features[:])
    # subFrameGen(data_frame, data_frame_features[:])
    plot_graph(data_frame, data_frame_features[:])
    # print(data_frame)


start_question1("iris.data.txt")
