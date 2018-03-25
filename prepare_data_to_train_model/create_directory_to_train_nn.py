import os
import shutil

base_dir = '../data/data_for_nn'
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

number_of_train = 0
number_of_val = 0

with open("../data/cars_data.csv") as cars_data:
    for line in cars_data:
        line = line.strip()
        splited_line = line.split(',')
        if splited_line[0] == 'TRAIN':
            path_to_file = splited_line[1]
            label = splited_line[2]
            label = label.replace('/', '*')
            path_to_label = os.path.join(train_dir, label)
            if not os.path.isdir(path_to_label):
                os.mkdir(path_to_label)
                os.mkdir(os.path.join(validation_dir, label))
            shutil.copy(path_to_file, path_to_label)
            number_of_train += 1
        elif splited_line[0] == 'VALIDATION':
            path_to_file = splited_line[1]
            label = splited_line[2]
            label = label.replace('/', '*')
            path_to_label = os.path.join(validation_dir, label)
            shutil.copy(path_to_file, path_to_label)
            number_of_val += 1
print(number_of_train)
print(number_of_val)
