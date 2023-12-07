import matplotlib.pyplot as plt
import numpy as np

# Function to read the .txt file and extract loss values
def read_loss_values(file_path):
    loss = []
    with open(file_path, 'r') as file:
        contents = file.readlines()
        for i in contents:
            loss_values = i.split(' ')[1]
            loss.append(float(loss_values))
    return loss

# Replace 'your_file.txt' with the path to your actual file
file_path = 'train_loss_log.txt'

# Read loss values from the file
loss_values = read_loss_values(file_path)

# Create an array for the time steps (assuming each loss value corresponds to a time step)
time_steps = np.arange(len(loss_values))
time_steps = time_steps[:] * 100

# Plotting the loss values
plt.figure(figsize=(10, 6))
plt.xlim(0,900000)
plt.yscale("log")
plt.plot(time_steps, loss_values, marker='o', linestyle='-')
plt.title('Loss Value Changes Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.grid(False)
plt.savefig('test.png')