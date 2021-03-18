import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus

# csv file -> dataframe
df = pd.read_csv('wineNormalized.csv')

# Target Feature -> Y & descriptive features -> X
x, y = df.iloc[:, :-1].values, df['Class'].values

# dataframe -> training/testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.33, random_state=0)

df.pop('Class')

# numpy arrays -> dataframes
df_train = pd.DataFrame(x_train, columns=df.columns)
df_train['Class'] = y_train
df_test = pd.DataFrame(x_test, columns=df.columns)
df_test['Class'] = y_test

# datasets -> csv files
df_train.to_csv('train_dataset.csv', index=False)
df_test.to_csv('test_dataset.csv', index=False)

resultsEntropy = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    resultsEntropy.loc[i] = [i, train_score, test_score]
    
resultsEntropy.pop('LevelLimit')
resultsEntropy.plot()
plt.title('Entropy Decision Tree Score Chart')
plt.xlabel("tree depth level")
plt.ylabel("% accuracy")
plt.show()

resultsGini = pd.DataFrame(columns=['LevelLimit', 'Training Score', 'Testing Score'])
for i in range(1, 15):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=0)
    clf = clf.fit(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    train_score = clf.score(x_train, y_train)
    resultsGini.loc[i] = [i, train_score, test_score]
   
resultsGini.pop('LevelLimit')
resultsGini.plot()
plt.title('Gini Decision Tree Score Chart')
plt.xlabel("tree depth level")
plt.ylabel("% accuracy")
plt.show()

clf = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=0)
clf = clf.fit(x_train, y_train)

dot_data = export_graphviz(clf,
                           feature_names=df.columns,
                           out_file=None,
                           filled=True,
                           rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('tree.png')