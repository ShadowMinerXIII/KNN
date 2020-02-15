import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

f = open("D:\Master's work\AI\Pima.txt", "r")
features = []
targets = []
allLines = f.readlines()

for line in allLines:
    line = line.strip().split('\t')
    X = line[:-1]
    X = [float(i) for i in X]
    y = int(line[-1:][0])
    features.append(X)
    targets.append(y)
    
f.close()
features = np.array(features)
targets = np.array(targets)
indices = np.random.permutation(len(features))
features_train = features[indices[:-100]]
targets_train = targets[indices[:-100]]
features_test = features[indices[-100:]]
targets_test = targets[indices[-100:]]

scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(features_train, targets_train)
predictions = knn.predict(features_test)
numTesting = features_test.shape[0]
numCorrect = (targets_test == predictions).sum()
accuracy = float(numCorrect) / float(numTesting)
print("No. correct={0}, No. testing examples={1}, prediction accuracy={2} per cent".format(numCorrect, numTesting, round(accuracy*100, 2)))
