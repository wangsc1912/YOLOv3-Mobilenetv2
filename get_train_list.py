import os

# data_dir = 'dataset/img/'
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, 'dataset/img')
file_name_list = os.listdir(data_dir)
a = file_name_list[-1]
with open(os.path.join(root, 'trainList.part'), 'w') as f:
    for name in file_name_list:
        f.write('/' + name + '\n')
