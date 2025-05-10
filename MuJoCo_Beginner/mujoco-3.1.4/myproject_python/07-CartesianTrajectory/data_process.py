import csv
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
        des_x.append(float(row[1]))
        x.append(float(row[2]))
        des_z.append(float(row[3]))
        z.append(float(row[4]))


plt.subplot(211)
plt.title("x, z")
plt.grid()
plt.plot(t, x)
plt.plot(t, z)
plt.legend(["x", "z"])
plt.subplot(212)
plt.title("xz")
plt.grid()
plt.plot(des_x, des_z)
plt.plot(x, z)
plt.legend(["desired", "measured"])


plt.show()

