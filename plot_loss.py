import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ptick 
import scienceplots

plt.style.use(['science','ieee'])

fig, ax = plt.subplots()

# Function to read the .txt file and extract loss values
def read_loss_values(file_path):
    loss = []
    l1, l2 = [], []
    with open(file_path, 'r') as file:
        contents = file.readlines()
        for i in contents:
            loss_values = i.split(' ')[1].replace(',','')
            #tmp_l1 = i.split(' ')[2].replace(',','')
            #tmp_l1 = tmp_l1.replace('l1:','')
            #tmp_l2 = i.split(' ')[3].replace(',','')
            #tmp_l2 = tmp_l2.replace('l2:','')
            loss.append(float(loss_values))
            #l1.append(float(tmp_l1))
            #l2.append(float(tmp_l2))
    return loss, l1, l2

# Replace 'your_file.txt' with the path to your actual file
file_path = 'train_loss_log_unsteady_algorithm_change.txt'

# Read loss values from the file
loss_values, l1_value, l2_value = read_loss_values(file_path)

# Create an array for the time steps (assuming each loss value corresponds to a time step)
time_steps = np.arange(len(loss_values))
#time_steps = time_steps[:] * 100

# Plotting the loss values
#plt.figure(figsize=(10, 6))
#plt.xlim(0,1000000)
plt.yscale("log")


ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
ax.ticklabel_format(style="sci", axis="x", scilimits=(3,3))   # 10^3（10の3乗）単位にする。
plt.plot(time_steps, loss_values)

#plt.xlim(-1.0,4100)
#diffs_l1 = np.diff(l1_value)
#diffs_l2 = np.diff(l2_value)

#plt.plot(time_steps, l1_value)
#plt.plot(time_steps, l2_value)

#plt.plot(diffs_l1)
#plt.plot(diffs_l2)
#print(diffs_l2)
#print(diffs_l1)
#print(diffs_l2)

#

plt.xlabel('Learning iteration [-]')
plt.ylabel('Loss value [-]')
plt.grid(False)
plt.legend()
plt.savefig('test.png')