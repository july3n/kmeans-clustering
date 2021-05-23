#!/usr/bin/env python
# coding: utf-8

# In[100]:


from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[101]:



Data = {'x': [25,34,22,27,23,24,31,22,35,26,28,54,57,43,36,27,29,52,32,47,39,48,35,33,44,45,38,43,41,46],
        'y': [79,51,53,78,99,92,73,57,69,75,51,32,40,77,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }


# In[102]:


df = DataFrame(Data,columns=['x','y'])


# In[103]:


kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)


# In[111]:


plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=50)
plt.title('Kişi Sonuç Değerlendirmesi')
plt.ylabel('Başarı Puanı')
plt.xlabel('Yaş ')
plt.show()

