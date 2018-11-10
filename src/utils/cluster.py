import numpy as np

class KMean:
    def __init__(self, n_clusters, distance_func,distance_func_context):
        self.n_clusters = n_clusters
        self.distance_func = distance_func
        self.distance_func_context = distance_func_context


    def fit(self,dataSet):
        numSamples = len(dataSet)
        # first column stores which cluster this sample belongs to,
        # second column stores the error between this sample and its centroid
        clusterAssment = np.mat(np.zeros((numSamples, 2)))
        clusterChanged = True

        ## step 1: init centroids
        centroids = self.__initCentroids(dataSet, self.n_clusters)

        while clusterChanged:
            clusterChanged = False
            ## for each sample
            for i in range(numSamples):
                minDist = 100000.0
                minIndex = 0
                ## for each centroid
                ## step 2: find the centroid who is closest
                for j in range(self.n_clusters):
                    distance = self.distance_func(centroids[j], dataSet[i],self.distance_func_context)
                    if distance < minDist:
                        minDist = distance
                        minIndex = j

                ## step 3: update its cluster
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    clusterAssment[i, :] = minIndex, minDist ** 2

            ## step 4: update centroids
            for j in range(self.n_clusters):
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                #centroids[j, :] = np.mean(pointsInCluster, axis=0)
                centroids[j] = pointsInCluster

        return centroids, clusterAssment


    def __initCentroids(self,dataSet, k):
        numSamples, dim = len(dataSet),1
        centroids = []
        for i in range(k):
            index = int(np.random.uniform(0, numSamples))
            centroids.append(dataSet[index])
        return centroids


