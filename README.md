One cool application of Principal Component Analysis is Eigenfaces, in which we apply PCA to reduce the dimensions of a set of face images. This can end up being used to search for a matching in a large facial dataset (e.g. the casino black list). 

<img width="764" alt="Eigenfaces" src="https://github.com/ggcr/eigenfaces-pca/assets/57730982/50b78bd9-3acb-4b90-ae9f-0f541057c734">

We define first a matrix $A$ with $400$ facial gray-scale pictures of size $64\times 64$.

<p align='center'>
$A_{400\times4096}\;, \quad(faces)$
</p>

Then we ''center'' that data with the mean-shift operation, by taking the mean $\mu$ of each row:

<p align='center'>
$B = A - \mu_A\;,$
</p>

Now, we can compose our covariance matrix $S$ that will represent the correlation between all the variables of our data:

<p align='center'>
$S = B^T B\;,$
</p>

Now, we have a real symmetric matrix, that is, by the way covariances are expressed, we only have the first half of the matrix of unique data, the other half will be a mirror of the aforementioned data, this allows us to find for Eigenvectors and Eigenvalues with a certainty that they exist:

<p align='center'>
$\vec{\lambda},\,\vec{V} = \texttt{torch.lineal.eig(}S\texttt{)}\;,$
</p>

Note that, the \texttt{PyTorch} library already sorted the Eigenvectors by the quantity of information retrieved in the Eigenvalues, and we can do a first analysis of the situation by calculing the Total Variance of the dataset ($T$), and how much of the total variance is represented in each of the Principal Components. The first Principal Component represents a $23.81\%$ of the total information.

<p align='center'>
$T=Tr(\vec{\lambda})=79.11\;,\quad \frac{\vec{\lambda_1}}{T} = 23.81\%\;,$
</p>

Now we will keep the first $k$ Principal Components and we will perform the PCA reconstruction by building our Projection matrices $P$.

<p align='center'>
$Z = \vec{V}\!(:\,,\,:\!k)\;,$
</p>

<p align='center'>
$P = B Z Z^{T}\;.$
</p>
