#%%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

#%%
plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
#%%
#plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("***KMeans")
plt.show()
#%%
# Elbow metod

mms = MinMaxScaler()
mms.fit(X)
X_scaled = mms.transform(X)

sum_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters = k).fit(X_scaled)
    sum_squared_distances.append(km.inertia_)

plt.plot(K, sum_squared_distances,'bx-')
plt.title("Elbow Method")
plt.xlabel("nombre de clusters")
plt.ylabel("Somme des carres des distances")
plt.show() 

#%%