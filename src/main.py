import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.datasets import fetch_olivetti_faces

faces, _ = fetch_olivetti_faces(data_home='../data/', return_X_y=True, shuffle=True, random_state=np.random.RandomState(0))
faces = torch.from_numpy(faces)

n = faces.shape[0]

A = torch.empty((n, 64 * 64), dtype=torch.float32)
for i in range(n):
    A[i, :] = torch.flatten(faces[i, :])

mean_A = torch.mean(A, 0, keepdims=True)

# mean-shifted data matrix X
X = A - mean_A

# covariance matrix S
S = (1 / (n - 1)) * (X.t() @ X)

# eigenvalues and eigenvectors (already sorted)
eigenvalues, eigenvectors = torch.linalg.eig(S)
eigenvectors = eigenvectors.float()
eigenvalues = eigenvalues.float()

# total variance
T = torch.sum(eigenvalues)
print("T = {}".format(T))

# variance of first principal component
lambda_1 = eigenvalues[0] / T
print("lambda_1 / T = {}".format(lambda_1))

# variance of second principal component
lambda_2 = eigenvalues[1] / T
print("lambda_2 / T = {}".format(lambda_2))

# discard n - k non-principal components
k = 50
eigenvectors = eigenvectors[:, :k]

sum = 0
for i in range(k):
    sum += eigenvalues[i] / T
print("Variance for k = {}: {}".format(k, sum))

# PCA reconstruction
P = X @ eigenvectors # projection matrix
P = P @ eigenvectors.t()
P += mean_A

# display and compute rmse
for i in range(n):
    image = A[i, :]
    image = torch.reshape(image, (64, 64))
    image = transforms.ToPILImage()(image)
    # image.show()

    image = P[i, :]
    image = torch.reshape(image, (64, 64))
    image = transforms.ToPILImage()(image)
    image.show()

    rmse = torch.sqrt(torch.mean((A[i, :] - P[i, :]) ** 2))
    print("RMSE = {}".format(rmse))

    input()