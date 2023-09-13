import numpy as np

# Generate random indices for a dataset subsample of size 5000
total_samples = 145585  # Assuming you have a total of 10000 samples
subsample_size = 5000
random_indices = np.random.choice(total_samples, subsample_size, replace=False)

# Write the random indices to a text file
with open("random_indices4.txt", "w") as f:
    for index in random_indices:
        f.write(str(index) + "\n")