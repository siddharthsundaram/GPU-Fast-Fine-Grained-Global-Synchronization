import os
import random

def generate_random_input(file_name, min_size, max_size, min_increments, max_increments):
    """
    Generates a random input file with the specified format:
    - First line: size of the 1D array (random between min_size and max_size)
    - Second line: number of increments for each index (random between min_increments and max_increments)
    """
    array_size = random.randint(min_size, max_size)
    num_increments = random.randint(min_increments, max_increments)

    with open(file_name, 'w') as f:
        f.write("{}\n".format(array_size))
        f.write("{}\n".format(num_increments))

    return array_size, num_increments

def main():
    # Configuration for random generation
    output_folder = "inputs"  # Folder to store the files
    num_files = 10            # Number of files to generate
    min_size = 1              # Minimum array size
    max_size = 100            # Maximum array size
    min_increments = 1        # Minimum number of increments
    max_increments = 50       # Maximum number of increments

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate the files
    for i in range(num_files):
        file_name = os.path.join(output_folder, "test{}".format(i))
        array_size, num_increments = generate_random_input(file_name, min_size, max_size, min_increments, max_increments)
        # print "Generated '{}' with:".format(file_name)
        # print "  Array size: {}".format(array_size)
        # print "  Number of increments: {}".format(num_increments)

if __name__ == "__main__":
    main()