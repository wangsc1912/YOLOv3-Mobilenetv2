import os

# data_dir = 'dataset/img/'
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, 'dataset/img')
file_name_list = os.listdir(data_dir)

with open('trainList.part', 'w') as f:
    for name in file_name_list:
        f.write('/' + name + '\n')
