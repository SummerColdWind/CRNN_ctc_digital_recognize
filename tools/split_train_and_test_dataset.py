import os
import random
import shutil

dataset_dir_path = '../data/image/'
train_save_dir_path = '../data/train/'
test_save_dir_path = '../data/test/'

images = os.listdir(dataset_dir_path)
train_dataset = random.sample(images, int(len(images) * 0.8))
test_dataset = [image for image in images if image not in train_dataset]

if not os.path.exists(train_save_dir_path):
    os.mkdir(train_save_dir_path)

if not os.path.exists(test_save_dir_path):
    os.mkdir(test_save_dir_path)

with open('./data/label.txt', 'r') as label, \
        open('./data/train/label.txt', 'w') as train_label, \
        open('./data/test/label.txt', 'w') as test_label:
    for line in label.readlines():
        filename, num = line.strip().split('\t')[0], line.strip().split('\t')[1]
        num = int(num)
        new_line = filename + '\t' + str(num) + '\n'
        if filename in train_dataset:
            train_label.write(new_line)
        else:
            test_label.write(new_line)

for image in train_dataset:
    shutil.copy(dataset_dir_path + image, train_save_dir_path + image)

for image in test_dataset:
    shutil.copy(dataset_dir_path + image, test_save_dir_path + image)
