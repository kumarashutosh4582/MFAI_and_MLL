#Finding 
X = [12.1,13.2,15.6,17.2,18.8,10.3,11.7,16.4]
Y = [48,54,32,18,41,32,31,30]
Z = [101,171,112,132,140,112,151,96]

sum = 0
for i in range(len(X)):
    sum += X[i]
meanX = sum / len(X)

sum = 0
for i in range(len(Y)):
    sum += Y[i]
meanY = sum / len(Y)

sum = 0
for i in range(len(Z)):
    sum += Z[i]
meanZ = sum / len(Z)


# variance X
sum = 0
for i in range(len(X)):
    sum += (X[i] - meanX) ** 2
varX = sum / len(X)


# variance Y
sum = 0
for i in range(len(Y)):
    sum += (Y[i] - meanY) ** 2
varY = sum / len(Y)


# variance Z
sum = 0
for i in range(len(Z)):
    sum += (Z[i] - meanZ) ** 2
varZ = sum / len(Z)


# covariance XY
sum = 0
for i in range(len(X)):
    sum += (X[i] - meanX) * (Y[i] - meanY)
covXY = sum / len(X)


# covariance XZ
sum = 0
for i in range(len(X)):
    sum += (X[i] - meanX) * (Z[i] - meanZ)
covXZ = sum / len(X)


# covariance YZ
sum = 0
for i in range(len(Y)):
    sum += (Y[i] - meanY) * (Z[i] - meanZ)
covYZ = sum / len(Y)


ans = [
    [varX, covXY, covXZ],
    [covXY, varY, covYZ],
    [covXZ, covYZ, varZ]
]

print("Mean X =", meanX)
print("Mean Y =", meanY)
print("Mean Z =", meanZ)

print("Covariance Matrix")

for i in range(len(ans)):
    print(ans[i])


#Assignment 3/4 displaying eigen vector and eigen values

import numpy as np
eigenvalues, eigenvectors = np.linalg.eig(ans)
print("\nEigen Values")
for i in range(len(eigenvalues)):
    print(f"\tλ{i+1} = ",eigenvalues[i])
print("\nEigenVectors are :\n ",eigenvectors)