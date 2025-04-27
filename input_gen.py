import os
import random

def generate_random_input(file_name, min_size, max_size, min_increments, max_increments):
    array_size = random.randint(min_size, max_size)
    num_increments = random.randint(min_increments, max_increments)

    with open(file_name, 'w') as f:
        f.write("{}\n".format(array_size))
        f.write("{}\n".format(num_increments))

    return array_size, num_increments

def main():
    output_folder = "inputs"
    num_files = 10
    min_size = 1
    max_size = 100
    min_increments = 1
    max_increments = 50

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_files):
        file_name = os.path.join(output_folder, "test{}".format(i))
        array_size, num_increments = generate_random_input(file_name, min_size, max_size, min_increments, max_increments)

if __name__ == "__main__":
    main()