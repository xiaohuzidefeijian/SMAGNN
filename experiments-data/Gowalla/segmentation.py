import random

with open('ratings_raw.txt', 'r') as file:
    lines = file.readlines()

random.seed(42)

for i in range(1, 6):
    random.shuffle(lines)

    num_total = len(lines)
    num_train = int(0.9 * num_total)
    num_val = int(0.02 * num_total)

    training_set = lines[:num_train]
    validation_set = lines[num_train:num_train + num_val]
    testing_set = lines[num_train + num_val:]

    # 保存数据集到相应文件
    with open(f'../ml-32m-{i}_training.txt', 'w') as train_file:
        train_file.writelines(training_set)

    with open(f'../ml-32m-{i}_validation.txt', 'w') as val_file:
        val_file.writelines(validation_set)

    with open(f'../ml-32m-{i}_testing.txt', 'w') as test_file:
        test_file.writelines(testing_set)

print("Done.")
