# Load the required library and data
library(ISLR)
data("USArrests")

# Compute the distance matrix
dist_matrix <- dist(USArrests)

# Perform hierarchical clustering using complete linkage
hc_complete <- hclust(dist_matrix, method = "complete")

# Plot the dendrogram
plot(hc_complete, main = "Hierarchical Clustering with Complete Linkage", xlab = "", sub = "", cex = 0.9)