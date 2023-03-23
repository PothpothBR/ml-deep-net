def min_max(data, column):
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())     
    return data
