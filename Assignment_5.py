import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("virus_bacteria_data.csv")

viral = data[data["Class"] == "Virus"]
bacteria = data[data["Class"] == "Bacteria"]

#mean for both class
mean_viral = np.array([
    viral["CRP"].mean(),
    viral["Temperature"].mean()
])

mean_bacteria = np.array([
    bacteria["CRP"].mean(),
    bacteria["Temperature"].mean()
])

# calculating W_viral
W_viral = np.array([
    [np.var(viral["CRP"], ddof=1),
     np.cov(viral["CRP"], viral["Temperature"])[0][1]],

    [np.cov(viral["CRP"], viral["Temperature"])[0][1],
     np.var(viral["Temperature"], ddof=1)]
])


# calculating W_Bact
W_bact = np.array([
    [np.var(bacteria["CRP"], ddof=1),
     np.cov(bacteria["CRP"], bacteria["Temperature"])[0][1]],

    [np.cov(bacteria["CRP"], bacteria["Temperature"])[0][1],
     np.var(bacteria["Temperature"], ddof=1)]
])


# calculating [T]
T = np.array([
    [np.var(data["CRP"], ddof=1),
     np.cov(data["CRP"], data["Temperature"])[0][1]],

    [np.cov(data["CRP"], data["Temperature"])[0][1],
     np.var(data["Temperature"], ddof=1)]
])


# size of Classes
n1 = len(viral)
n2 = len(bacteria)


# calculating pooled within-class matrix [W]
W = ((n1 - 1) * W_viral + (n2 - 1) * W_bact) / (n1 + n2 - 2)

# between class cov Matrix
B = T - W

# S = B W^-1
S = np.linalg.inv(W) @ B

print(S)

# eigenvalue and eigenvector calculation
Evalue, Evector = np.linalg.eig(S)
idx = np.argmax(Evalue)
w = Evector[:, idx]

w1 = w[0]
w2 = w[1]

print(w1,w2)
z = -0.5 * np.dot(mean_viral + mean_bacteria, w)
x = np.linspace(data["CRP"].min(), data["CRP"].max(), 100)
y = -(w1 * x + z) / w2

print(z)


plt.scatter(viral["CRP"], viral["Temperature"], color="Green" , label = "Virus")
plt.scatter(bacteria["CRP"], bacteria["Temperature"], color="Red" , label = "Bacteria")
plt.title("Linear Discriminant Analysis")
plt.plot(x, y)

plt.legend()
plt.xlabel("CRP")
plt.ylabel("Temperature")

plt.ylim(33, 45)

plt.show()