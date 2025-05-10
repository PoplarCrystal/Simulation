import csv
import matplotlib.pyplot as plt


csv_path = "demo.csv"
t = []
q1 = []
q2 = []
with open(csv_path, "r") as file:
    reader = csv.reader(file)
    header_row = next(reader)  # clear the csv-title
    header_row = next(reader)  # clear the ["q1", "q2"]
    for row in reader:
        t.append(float(row[0]))
        q1.append(float(row[1]))
        q2.append(float(row[2]))

plt.subplot(211)
plt.title("q1")
plt.grid()
plt.plot(t, q1)
plt.subplot(212)
plt.title("q2")
plt.grid()
plt.plot(t, q2)


plt.show()

