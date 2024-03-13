import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ptick   # ライブラリをインポートする。
import scienceplots

plt.style.use(['science','ieee'])

# Function to read the .txt file and extract loss values
def read_loss_values(file_path):
    val_loss, c_loss, u_loss, v_loss, w_loss, p_loss = [],[],[],[],[],[]
    with open(file_path, 'r') as file:
        contents = file.readlines()
        for i in contents:
            line = i
            tmp_val_loss = i.split(',')[0].split(' ')[1]
            tmp_val_c_loss = i.split(',')[1].split(' ')[2]
            #tmp_val_u_loss = i.split(',')[2].split(' ')[2]
            #tmp_val_v_loss = i.split(',')[3].split(' ')[2]
            #tmp_val_w_loss = i.split(',')[4].split(' ')[2]
            #tmp_val_p_loss = i.split(',')[5].split(' ')[2]
            val_loss.append(float(tmp_val_loss))
            c_loss.append(float(tmp_val_c_loss))
            #u_loss.append(float(tmp_val_u_loss))
            #v_loss.append(float(tmp_val_v_loss))
            #w_loss.append(float(tmp_val_w_loss))
            #p_loss.append(float(tmp_val_p_loss))
    return val_loss, c_loss, u_loss, v_loss, w_loss, p_loss

# Replace 'your_file.txt' with the path to your actual file
file_path = 'val_err_log.txt'

# Read loss values from the file
val_loss, c_loss, u_loss, v_loss, w_loss, p_loss  = read_loss_values(file_path)

fig, ax = plt.subplots()

# Create an array for the time steps (assuming each loss value corresponds to a time step)
time_steps = np.arange(len(val_loss))
time_steps = time_steps[:] * 1000

# Plotting the loss values
#plt.figure(figsize=(10, 6))
plt.yscale("log")

#plt.plot(time_steps, val_loss, linestyle='-', label='val/loss')
ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))   # こっちを先に書くこと。
ax.ticklabel_format(style="sci", axis="x", scilimits=(3,3))   # 10^3（10の3乗）単位にする。
plt.plot(time_steps, c_loss, linestyle='-', label='c_loss')
#plt.plot(time_steps, val_loss, linestyle='-', label='val_loss')
#plt.plot(time_steps, u_loss, linestyle='-', label='u_loss')
#plt.plot(time_steps, v_loss, linestyle='-', label='v_loss')
#plt.plot(time_steps, w_loss, linestyle='-', label='w_loss')
#plt.plot(time_steps, p_loss, linestyle='-', label='p_loss')
#plt.title('Loss Value Changes Over Time')
plt.xlabel('Learning iteration [-]')
plt.ylabel('L2-error [-]')
#plt.xlim(0,500000)
#plt.grid(True)
#plt.legend()
plt.savefig('test_val.png')