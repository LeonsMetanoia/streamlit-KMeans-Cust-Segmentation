import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

# Rename columns for simplicity
df.rename(index=str, columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'Score'
}, inplace=True)

# Drop unnecessary columns
x = df.drop(['CustomerID', 'Gender'], axis=1)

# Display dataset in Streamlit
st.header('DataSet')
st.write(x)

# Calculate inertia for different cluster sizes to find elbow point
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=0).fit(x)
    clusters.append(km.inertia_)

# Plot the Elbow Method graph
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)

ax.set_title('Finding Elbow')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')

# Annotate possible elbow points
ax.annotate("Possible Elbow Point", xy=(3, clusters[2]), xytext=(3, clusters[2] + 40000), 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate("Possible Elbow Point", xy=(5, clusters[4]), xytext=(5, clusters[4] + 50000), 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

# Display the elbow plot in Streamlit
st.pyplot(fig)

# Sidebar input for number of clusters
st.sidebar.subheader("K Total Value")
clust = st.sidebar.slider("Pilih Jumlah Cluster:", 2, 10, 3, 1)

# Function to perform KMeans clustering and display results
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust, random_state=0).fit(x)
    x['Labels'] = kmean.labels_

    # Create scatter plot of the clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='Income', y='Score', hue='Labels', size='Labels', 
                    data=x, palette=sns.color_palette('hls', n_clust), sizes=(50, 200), ax=ax)

    # Add annotations for each cluster
    for label in x['Labels'].unique():
        ax.annotate(
            label, 
            (x[x['Labels'] == label]['Income'].mean(), 
             x[x['Labels'] == label]['Score'].mean()),
            horizontalalignment='center', verticalalignment='center',
            size=12, weight='bold', color='black'
        )

    # Display cluster plot in Streamlit
    st.header('Cluster Plot')
    st.pyplot(fig)
    st.write(x)

# Call the k_means function with the selected number of clusters
k_means(clust)
