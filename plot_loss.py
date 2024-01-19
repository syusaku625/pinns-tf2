import matplotlib.pyplot as plt
import numpy as np

# Function to read the .txt file and extract loss values
def read_loss_values(file_path):
    loss = []
    l1, l2 = [], []
    with open(file_path, 'r') as file:
        contents = file.readlines()
        for i in contents:
            loss_values = i.split(' ')[1].replace(',','')
            tmp_l1 = i.split(' ')[2].replace(',','')
            tmp_l1 = tmp_l1.replace('l1:','')
            tmp_l2 = i.split(' ')[3].replace(',','')
            tmp_l2 = tmp_l2.replace('l2:','')
            loss.append(float(loss_values))
            l1.append(float(tmp_l1))
            l2.append(float(tmp_l2))
    return loss, l1, l2

# Replace 'your_file.txt' with the path to your actual file
file_path = 'train_loss_log.txt'

# Read loss values from the file
loss_values, l1_value, l2_value = read_loss_values(file_path)

# Create an array for the time steps (assuming each loss value corresponds to a time step)
time_steps = np.arange(len(loss_values))
time_steps = time_steps[:] * 100

# Plotting the loss values
plt.figure(figsize=(10, 6))
#plt.xlim(0,900000)
plt.yscale("log")
plt.plot(time_steps, loss_values, marker='o', linestyle='-', markersize=1)
plt.plot(time_steps, l1_value, marker='x', linestyle='-', markersize=1)
plt.plot(time_steps, l2_value, marker='^', linestyle='-', markersize=1)
plt.title('Loss Value Changes Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.grid(False)
plt.savefig('test.png')