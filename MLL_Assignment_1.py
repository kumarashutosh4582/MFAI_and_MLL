import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./weight-height.csv")


def loss_function(m, b, points):
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].Height
        y = points.iloc[i].Weight

        total_error += (y - (m*x + b))**2

    return total_error / float(len(points))


def gradient_descent(m_now, b_now, points, LR):

    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):

        x = points.iloc[i].Height
        y = points.iloc[i].Weight

        error = y - (m_now*x + b_now)

        m_gradient += -(2/n) * x * error
        b_gradient += -(2/n) * error

    m = m_now - m_gradient * LR
    b = b_now - b_gradient * LR

    return m, b


L = 0.0001
m = 0
b = 0
epoch = 500

for i in range(epoch):

    if i % 50 == 0:
        print(f"Iteration Cycle {i} , Loss = {loss_function(m,b,data)}")

    m, b = gradient_descent(m, b, data, L)


print("m =", m)
print("b =", b)


plt.scatter(data.Height, data.Weight, color="Red")

plt.plot(
    range(60, 80),
    [(m*x) + b for x in range(60,80)]
)
plt.title("Linear Regression")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()