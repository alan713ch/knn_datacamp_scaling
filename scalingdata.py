# alan izar
# scaling data to combat noise
# from hugo bowne-anderson in data camp

# generating the clustered data
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

n_samples = 2000
X, y = make_blobs(n_samples, centers=4, n_features=2, random_state=0)  # gaussian blobs for clustering
# X: generated samples, y: integer labels for cluster membership

# time to plot the data!
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)  # first plot, scatter of the generated data
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
plt.subplot(1, 2, 2)  # second plot, histogram of the labels. They should all be equal
plt.hist(y)
#plt.show()

#extract the data from X as a dataframe
df = pd.DataFrame(X)
pd.DataFrame.hist(df, figsize=(20,5))
plt.show()

#train and test splits
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #why 0.2, why 42?

#plotting time!
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('Training Set')
plt.scatter(X_train[:,0], X_train[:,1], c=y_train,alpha=0.7)
plt.subplot(1,2,2)
plt.scatter(X_test[:,0], X_test[:,1], c= y_test, alpha=0.7)
plt.title('Test Set')
plt.show()

#now let's use the knn method!
from sklearn import neighbors, linear_model
knn = neighbors.KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)

#let's look at how ACCURATE the model is for the test and training set. Remember, this is only accuracy
print('k-NN score for test set: %f' % knn_model.score(X_test, y_test))
print('k-NN score for train set: %f' % knn_model.score(X_train, y_train))

#other metrics
from sklearn.metrics import classification_report
y_true, y_pred = y_test, knn_model.predict(X_test)
print (classification_report(y_true,y_pred))

#now once more, with scaling
from sklearn.preprocessing import scale
Xs = scale(X)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

#plotting time!
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title('Scaled Training Set')
plt.scatter(Xs_train[:,0], Xs_train[:,1], c=ys_train,alpha=0.7)
plt.subplot(1,2,2)
plt.scatter(Xs_test[:,0], Xs_test[:,1], c= ys_test, alpha=0.7)
plt.title('Scaled Test Set')
plt.show()

#now let's use the knn method!
knn_model_s = knn.fit(Xs_train, ys_train)

#let's look at how ACCURATE the model is for the test and training set. Remember, this is only accuracy
print('k-NN score for scaled test set: %f' % knn_model.score(Xs_test, ys_test))
print('k-NN score for scaled train set: %f' % knn_model.score(Xs_train, ys_train))

#other metrics
ys_true, ys_pred = ys_test, knn_model.predict(Xs_test)
print (classification_report(ys_true,ys_pred))

#there's not much of a difference when scaling becase the scores were pretty close already. Use scaling when they differ wildly

#since this is an example case, we'll add noise to see its effect on the k-NN model (and how beneficial can scaling be!)

ns = 10**(3) #strength of the noise, in the thousands
newcol = np.transpose([ns*np.random.randn(n_samples)]) #new column with the random numbers, generated from a randon generator based
#on the amount of samples, multiplied to the thousands so it is in the same order of magnitude
Xn = np.concatenate((X,newcol), axis= 1) #concatenate the noise to the data

#plotting time!
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111,projection = '3d', alpha= 0.5)
ax.scatter(Xn[:,0],Xn[:,1],Xn[:,2], c= y)
plt.show()

#once more, with noise
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, y, test_size=0.2, random_state=42)
knn = neighbors.KNeighborsClassifier()
knn_model = knn.fit(Xn_train, yn_train)
print('k-NN score for noisy train set: %f' % knn_model.score(Xn_test, yn_test))

#scaling so it becomes a better model
Xns = scale(Xn)
s = int(.2*n_samples)
Xns_train = Xns[s:]
yns_train = y[s:]
Xns_test = Xns[:s]
yns_test = y[:s]
knn = neighbors.KNeighborsClassifier()
knn_models = knn.fit(Xns_train, yns_train)
print('k-NN score for test set: %f' % knn_models.score(Xns_test, yns_test))

#we are going to check for accuracy now, and since the code is repetitive, function time!
def accu(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    knn = neighbors.KNeighborsClassifier()
    knn_model = knn.fit(X_train, y_train)
    return(knn_model.score(X_test,y_test))
#it's the same code as before, but now it is all neatly wrapped up in one line!

noise = [10**i for i in np.arange(-1,6)] #we are generating the noise in different orders of magnitude, from -1 to 5
A1 = np.zeros(len(noise))
A2 = np.zeros(len(noise))
count = 0

for ns in noise:
    newcol = np.transpose([ns*np.random.randn(n_samples)])
    Xn = np.concatenate((X,newcol),axis=1)
    Xns = scale(Xn)
    A1[count] = accu(Xn,y)
    A2[count] = accu(Xns,y)
    count +=1

#plotting time!
plt.scatter(noise, A1)
plt.plot(noise, A1, label = 'unscaled', linewidth=2)
plt.scatter(noise, A2, c='r') #color red for the scaled
plt.plot(noise, A2, label = 'scaled', linewidth=2)
plt.xscale('log') #logarithmic scale
plt.xlabel('noise strength')
plt.ylabel('accuracy')
plt.legend(loc=3)
plt.show()