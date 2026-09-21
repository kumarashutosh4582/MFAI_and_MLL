import matplotlib.pyplot as plt

Height = [1.70, 1.62, 1.52, 1.85, 1.91, 1.42]
Weight = [72, 64, 84, 80, 72, 70]

# Mean of Height
sum = 0

for i in range(len(Height)):
    sum += Height[i]

meanHeight = sum / len(Height)

# Mean of Weight
sum = 0

for i in range(len(Weight)):
    sum += Weight[i]

meanWeight = sum / len(Weight)

print("Mean Height:", meanHeight)
print("Mean Weight:", meanWeight)


# Variance of Height
sum = 0

for i in range(len(Height)):
    sum += (Height[i] - meanHeight) ** 2

varHeight = sum / len(Height)


# Variance of Weight
sum = 0

for i in range(len(Weight)):
    sum += (Weight[i] - meanWeight) ** 2

varWeight = sum / len(Weight)


# Covariance of Height and Weight
sum = 0

for i in range(len(Height)):
    sum += (Height[i] - meanHeight) * (Weight[i] - meanWeight)

covHeightWeight = sum / len(Height)


# 2 x 2 Covariance Matrix
ans = [
    [varHeight, covHeightWeight],
    [covHeightWeight, varWeight]
]

print("Variance of Height:", varHeight)
print("Variance of Weight:", varWeight)
print("Covariance:", covHeightWeight)

print("2 x 2 Covariance Matrix:")
for row in ans:
    print(row)


# Scatter Plot
plt.scatter(Height, Weight)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight")

plt.show()