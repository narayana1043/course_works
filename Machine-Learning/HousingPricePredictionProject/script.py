from datareader import read_data as ip_read

train, test = ip_read()
for col in list(train.columns.values):
    print(col, train[col].unique()[:10])
print(list(train.columns.values))