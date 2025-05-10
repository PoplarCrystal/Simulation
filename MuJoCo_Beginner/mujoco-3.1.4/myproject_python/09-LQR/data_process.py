import csv
import numpy as np
import matplotlib.pyplot as plt


csv_path = "demo.csv"
t = []
des_x, des_z = [], []
x, z = [], []
with open(csv_path, "r") as file:
    reader = csv.reader(file)
    header_row = next(reader)  # clear the csv-title
    header_row = next(reader)  # clear the ["q1", "q2"]
    for row in reader:
        t.append(float(row[0]))
        x.append(float(row[1]))
        z.append(float(row[2]))

des_x = [0]*len(x)
des_z = [0]*len(z)

plt.subplot(211)
plt.title("q1")
plt.grid()
plt.plot(t, des_x)
plt.plot(t, x)
plt.legend(["desired", "measured"])
plt.subplot(212)
plt.title("q2")
plt.grid()
plt.plot(t, des_z)
plt.plot(t, z)
plt.legend(["desired", "measured"])


plt.show()

