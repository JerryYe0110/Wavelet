import matplotlib.pyplot as plt

# 定义 x 坐标
x = [1, 2, 3, 4, 5]

# 定义 y 坐标
y1 = [3, 6, 2, 7, 1]
y2 = [4, 1, 8, 3, 5]
y3 = [2, 3, 4, 2, 6]

Restormer = [1.52, 4.9]
NAFNet = [0.3, 1.0, 4.1]
ours = [0.26, 0.54, 1.29]


Restormer_y = [38.96, 39.637]
NAFNet_y = [39.213, 39.645, 39.81]
ours_y = [39.233, 39.578, 39.728]

plt.plot(Restormer, Restormer_y, marker='o', color='blue', label='Restormer')
plt.plot(NAFNet, NAFNet_y,  marker='s', color='red', label='NAFNet')
plt.plot(ours, ours_y, marker='*', color='green', label='ours')

plt.grid(True)

plt.xlabel('MACs(G)')
plt.ylabel('PSNR on SIDD')
plt.legend(loc='lower right')

plt.show()

