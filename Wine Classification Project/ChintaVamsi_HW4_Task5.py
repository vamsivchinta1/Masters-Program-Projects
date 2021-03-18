import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Loading the csv file in a dataframe
df_train = pd.read_csv('train_dataset.csv')
df_test = pd.read_csv('test_dataset.csv')



top_10_features = [ 'Color intensity', 'Flavanoids', 'Alcohol','Total phenols',] #'Proline','Magnesium', 'Nonflavanoid phenols']#, 'Malic acid', 'Ash', 'OD280/OD315 of diluted wines','Alcalinity of ash', 'Hue']



df_train_topfeatures = df_train[top_10_features]
df_test_topfeatures = df_test[top_10_features]

df_train_topfeatures['Proline + Magnesium'] = ((df_train['Proline'] + df_train['Magnesium'])/2)
df_test_topfeatures['Proline + Magnesium'] = ((df_test['Proline'] + df_test['Magnesium'])/2)

x_train, y_train = df_train_topfeatures.iloc[:, :].values, df_train['Class'].values
x_test, y_test = df_test_topfeatures.iloc[:, :].values, df_test['Class'].values

scores = {}
results = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    results.loc[i] = [i, train_score, test_score]

results.pop('LevelLimit')
results.plot()
plt.title('Decision Tree with Select Features + Derived Feature')
plt.xlabel('Max Tree Depth Level')
plt.ylabel('% Accuracy')
plt.show()

df_train['Proline + Magnesium'] = ((df_train['Proline'] + df_train['Magnesium'])/2)
df_test['Proline + Magnesium'] = ((df_test['Proline'] + df_test['Magnesium'])/2)

y_train_temp = df_train.pop('Class')
y_test_temp = df_test.pop('Class')

x_train, y_train = df_train.iloc[:, :].values, y_train_temp.values
x_test, y_test = df_test.iloc[:, :].values, y_test_temp.values

resultsDerived = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    resultsDerived.loc[i] = [i, train_score, test_score]

resultsDerived.pop('LevelLimit')
resultsDerived.plot()
plt.title('Decision Tree Results from Derived Features')
plt.show()

# Joining Top 10 Features with the new Derived ones
df_train_latest = df_train_derived.join(df_train_topfeatures)
df_test_latest = df_test_derived.join(df_train_topfeatures)

X_train, y_train = df_train_latest.iloc[:, :].values, df_train['Class'].values
X_test, y_test = df_test_latest.iloc[:, :].values, df_test['Class'].values

resultsMixed = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    train_score = clf.score(X_train, y_train)
    resultsMixed.loc[i] = [i, train_score, test_score]

resultsMixed.pop('LevelLimit')
resultsMixed.plot()
plt.title('Decision Tree Results from Derived and Top Features')
plt.show()


