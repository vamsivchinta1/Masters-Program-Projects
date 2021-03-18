import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Loading the csv file in a dataframe
df_train = pd.read_csv('train_dataset.csv')
df_test = pd.read_csv('test_dataset.csv')

# dataframe -> training and test datasets
x_train, y_train = df_train.iloc[:, :-1].values, df_train['Class'].values
x_test, y_test = df_test.iloc[:, :-1].values, df_test['Class'].values

# euclidean - uniform (e1)
k_e1 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
avg_train_e1 = []
avg_test_e1 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', p=2, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_e1.loc[i] = [i, train_score, test_score]

    avg_train_e1.append(train_score)
    avg_test_e1.append(test_score)

avg_trainscore_e1 = sum(avg_train_e1)/len(avg_train_e1)
print('average train score for e1 =', avg_trainscore_e1)

avg_testscore_e1 = sum(avg_test_e1)/len(avg_test_e1)
print('average test score for e1 =', avg_testscore_e1)


# euclidean - weighted distance (e2)
k_e2 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
x_e2 = []
y_e2 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', p=2, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_e2.loc[i] = [i, train_score, test_score]
    x_e2.append(test_score)
    y_e2.append(train_score)

mean_y_e2 = sum(y_e2)/len(y_e2)
print('mean training test score for e2 is', mean_y_e2)

mean_e2 = sum(x_e2)/len(x_e2)
print(mean_e2)

# manhattan - uniform
k_m1 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
x_m1 = []
y_m1 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform', p=1, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_m1.loc[i] = [i, train_score, test_score]
    x_m1.append(test_score)
    y_m1.append(train_score)

mean_y_m1 = sum(y_m1)/len(y_m1)
print('m1', mean_y_m1)

mean_m1 = sum(x_m1)/len(x_m1)
print(mean_m1)


# manhattan - weighted distance
k_m2 = pd.DataFrame(columns=['K neighbors', 'Training Score', 'Testing Score'])
x_m2 = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', p=1, metric='minkowski')
    knn.fit(x_train, y_train)

    test_score = knn.score(x_test, y_test)
    train_score = knn.score(x_train, y_train)

    k_m2.loc[i] = [i, train_score, test_score]
    x_m2.append(test_score)

mean_m2 = sum(x_m2)/len(x_m2)
print(mean_m2)


# plotting
# euclidean - uniform
k_e1.pop('K neighbors')
k_e1.plot()
plt.title('KNN Euclidean Uniform Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()

# euclidean - weighted distance
k_e2.pop('K neighbors')
k_e2.plot()
plt.title('KNN Euclidean Weighted-Distance Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()

# manhattan - uniform
k_m1.pop('K neighbors')
k_m1.plot()
plt.title('KNN Manhattan Uniform Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()

# manhattan - weighted distance
k_m2.pop('K neighbors')
k_m2.plot()
plt.title('KNN Manhattan Weighted-Distance Score Chart')
plt.ylabel('% Accuracy')
plt.xlabel('k value')
plt.show()