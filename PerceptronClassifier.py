import numpy as np;
# reference to Tao's extra homework a6
class Perceptron_Classifer:
    # multiple do the multiplication for two vectors by numpy
    def multiple(self,w,x):
        return np.dot(w,x)
    #algorithm perfroms perceptron learning algorithm
    def algorithm (self,train_points, train_labels):
        # an example of points will be [  3.90000000e+01   7.75160000e+04   1.30000000e+01   2.17400000e+03
        #    0.00000000e+00   4.00000000e+01]
        # our initial w is (1,1,1,1,1,1,1,1,1,1)
        w=np.array([1,1,1,1,1,1,1,1,1,1])
        #coverge is checkpoint for whether the function is coverge and ready to output
        coverge=False
        length=len(train_points)
        # the following two sets are storing the number of the whole train point set
        classificationlist1=[]
        classificationlist0=[]
        for i in range(length):
            if train_labels[i]==1:
                classificationlist1.append(i)
            else:
                classificationlist0.append(i)
        while (not coverge):
            # for all points x belongs to M+(1)
            for i in classificationlist1:
                x=np.array(train_points[i])

                if self.multiple(w,x)<=0:
                    w=w+x
                else:
                    coverge=True
            # for all points x belongs to M-(0)
            for i in classificationlist0:
                x=np.array(train_points[i])
                if self.multiple(w,x)>0:
                    w=w-x
                else:
                    coverge=True
        return w
    #classify use w get from above to classify our current point
    def classify(self,point, train_points, train_labels):
        w=self.algorithm(train_points, train_labels)
        if self.multiple(w,point)>0:
            return 1
        else:
            return 0














