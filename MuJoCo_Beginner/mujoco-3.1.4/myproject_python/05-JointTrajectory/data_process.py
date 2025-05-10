import csv
import matplotlib.pyplot as plt


csv_path = "demo.csv"
t = []
q1_ref, q1 = [], []
q2_ref, q2 = [], []
with open(csv_path, "r") as file:
    reader = csv.reader(file)
    header_row = next(reader)  # clear the csv-title
    header_row = next(reader)  # clear the ["q1", "q2"]
    for row in reader:
        t.append(float(row[0]))
        q1_ref.append(float(row[1]))
        q1.append(float(row[2]))
        q2_ref.append(float(row[3]))
        q2.append(float(row[4]))

plt.subplot(211)
plt.title("q1")
plt.grid()
plt.plot(t, q1_ref)
plt.plot(t, q1)
plt.legend(["desired", "measured"])
plt.subplot(212)
plt.title("q2")
plt.grid()
plt.plot(t, q2_ref)
plt.plot(t, q2)
plt.legend(["desired", "measured"])


plt.show()

