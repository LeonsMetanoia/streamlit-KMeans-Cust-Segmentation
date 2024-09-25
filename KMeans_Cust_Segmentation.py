import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)': 'Score'
}, inplace=True)

x = df.drop(['CustomerID', 'Gender'], axis=1)

st.header('DataSet')
st.write(x)


# Menampilkan Panah Elbow
clusters = []
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize = (12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)

ax.set_title('Finding Elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')


# Panah Elbow
ax.annotate("Possible Elbow Point", xy=(3, 140000), xytext=(3,50000), xycoords='data', 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2 ))

ax.annotate("Possible Elbow Point", xy=(5, 80000), xytext=(5,150000), xycoords='data', 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2 ))

st.set_option('deprecation.showPyplotGlobalUse', False)

elbo_plot = st.pyplot()


st.sidebar.subheader("K Total Value")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_

    plt.figure(figsize=(10,8))
    sns.scatterplot(x='Income', y='Score', hue='Labels', size='Labels', 
                data=x, palette=sns.color_palette('hls', n_clust), sizes=(20, 200))

# Loop untuk menambahkan anotasi pada setiap label
    for label in x['Labels'].unique():
        plt.annotate(
            label, 
            (x[x['Labels'] == label]['Income'].mean(), 
             x[x['Labels'] == label]['Score'].mean()),
             horizontalalignment='center', verticalalignment='center',
            size=12, weight='bold', color='black'
    )

    st.header('Cluster Plot')
    st.pyplot()
    st.write(x)

k_means(clust)

