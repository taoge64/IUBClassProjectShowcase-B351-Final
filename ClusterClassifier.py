import math
import numpy as np
import random
# identify the centriod for cluster
#reference: https://mubaris.com/posts/kmeans-clustering/
#           https://blog.csdn.net/qq_40597317/article/details/80949123
#           https://towardsdatascience.com/kmeans-clustering-for-classification-74b992405d0a
#non-supervised
class Cluster_Classifier:
    # init function
    def __init__(self,k):
        "k is the number of centriod, define while implement"
        self.k=k
       # self.trainingpoint=trainingpoint
    # calculate euclidean distance between the point(x) and each centroid(c)
    # redefined from original since we have d-dimension attributes per point
    def euclidean_distance(self,c,x):
        num=0
        if (len(c)!=len(x)):
            print("it should never happened")
            pass
        else:
            for i in range(len(c)):
                num+= (c[i] - x[i]) * (c[i] - x[i])
        return math.sqrt(num)
    #compare function will give you which centroid this point nearby
    def compare (self,centroid_list,x):
        belonging=0
        # example when we have 2 centroids, for loop will run twice and then give either 0 or 1 ( which is nearer point)
        for i in range(len(centroid_list)):
            if self.euclidean_distance(centroid_list[belonging],x)>self.euclidean_distance(centroid_list[i],x):
                belonging=i
        return belonging
    # random identify k centriod as a list for cluster
    def define_centriod(self,trainingpoint):
        length_of_list=len(trainingpoint)
        centriod=[]
        for i in range (self.k):
            ran=random.randint(0,length_of_list-1)
            centriod.append(trainingpoint[ran])
        return centriod
    #centroid_list is the list of centroid we storaged (given by define_centroid)
    #centroid_storage is the storage for each new centroid (calculate by average)
    def moving_cluster(self,centroid_list,trainingpoint):
        centroid_storage=[]
        for i in range(len(centroid_list)):
            # example for init storage  for 2 centroids
            #centroid_storage.append([])

            centroid_storage.append(np.array(np.zeros(len(trainingpoint[i]))))
        for i in range(len(trainingpoint)):
            belonging=self.compare(centroid_list,trainingpoint[i])
            centroid_storage[belonging]=centroid_storage[belonging]+trainingpoint[i]
        return centroid_storage
    #algorithm is perform Cluster classifier
    def algorithm(self,trainingpoint):
        old_centriod=self.define_centriod(trainingpoint)
        coverge=False
        #only when centroids never change, we output the answer
        #converge is the situation when no points group to the others, here use the old centriod equal to new centriod to present centroid
        while not coverge:
            new_centroid=self.moving_cluster(old_centriod,trainingpoint)
            if np.array_equal(old_centriod[0],new_centroid[1]) and np.array_equal(old_centriod[1],new_centroid[0]):
                    coverge=True
            elif np.array_equal(old_centriod[0],new_centroid[0]) and np.array_equal(old_centriod[1],new_centroid[1]):
                    coverge=True
            else:
                coverge=False
            if not coverge:
                old_centriod=new_centroid
        return old_centriod
    # classify method to perform classfiction about which this point belong to (0,1)
    def classify(self,points,training_point):
        centroid_list=self.algorithm(training_point)
        return self.compare(centroid_list,points)








