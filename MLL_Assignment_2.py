import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./insurance_data.csv')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_function(m, b, points):

    total_error = 0
    n = len(points)

    for i in range(n):

        x = points.iloc[i].age
        y = points.iloc[i].will_buy_insurance

        prediction = sigmoid(m*x + b)
        total_error += -(y*np.log(prediction) + (1-y)*np.log(1-prediction))

    return total_error / n


def gradient_descent(m_now, b_now, points, LR):

    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):

        x = points.iloc[i].age
        y = points.iloc[i].will_buy_insurance

        prediction = sigmoid(m_now*x + b_now)

        error = prediction - y
        m_gradient += (1/n) * x * error
        b_gradient += (1/n) * error

    m = m_now - LR * m_gradient
    b = b_now - LR * b_gradient

    return m, b


L = 0.001
m = 0
b = 0

epoch = 50000

for i in range(epoch):

    if i % 50 == 0:
        print(
            f"Iteration {i}, Loss = {loss_function(m,b,data)}"
        )

    m, b = gradient_descent(m, b, data, L)


print("m =", m)
print("b =", b)



age = int(input("Enter age: "))
probability = sigmoid(m*age + b)
print("Probability:", probability)
if probability >= 0.5:
    print(f"Yes, person with age {age} will buy insurance")
else:
    print(f"No, person with age {age} will not buy insurance")

plt.scatter(data.age,data.will_buy_insurance,color="red")
plt.plot(
    range(0, 100),
    [sigmoid(m*x + b) for x in range(0,100)]
)
plt.title("Logistic Regression")
plt.xlabel("Age ")
plt.ylabel("Prob")
plt.show()