import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# csv -> dataframe
df_train = pd.read_csv('train_dataset.csv')
df_test = pd.read_csv('test_dataset.csv')

temp_train_class = df_train.pop('Class')
temp_test_class = df_test.pop('Class')

x_train, y_train = df_train.iloc[:, :].values, temp_train_class.values
x_test, y_test = df_test.iloc[:, :].values, temp_test_class.values

indexR = 1
results = pd.DataFrame(columns=['Size of Forest', 'Training Score', 'Testing Score'])
for i in range(1, 100, 10):
    forest = RandomForestClassifier(criterion='gini', n_estimators=i, random_state=0)
    forest.fit(x_train, y_train)
    test_score = forest.score(x_test, y_test)
    train_score = forest.score(x_train, y_train)
    results.loc[indexR] = [i, train_score, test_score]
    indexR += 1

SizeofForest = results.pop('Size of Forest')
results.plot()
plt.title('Random Forest Entropy Score Chart')
plt.xlabel('# of Bootstrapes (x10)')
plt.ylabel('% Accuracy')
plt.show()

results['Size of Forest'] = SizeofForest

# Entropy Ranking
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=20,
                                random_state=0)
forest.fit(x_train, y_train)

feat_labels = df_train.columns
feature_dict = {}
importances = forest.feature_importances_
for i in range(len(importances)):
    feature_dict[feat_labels[i]] = importances[i]

sorted_by_value = sorted(feature_dict.items(), key=lambda kv: kv[1], reverse=True)
print('Feature Rank:')
rank = 1
for k, v in sorted_by_value:
    print(rank, k, v)
    rank += 1

# Gini Ranking
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=20,
                                random_state=0)
forest.fit(x_train, y_train)

feat_labels = df_train.columns
feature_dict = {}
importances = forest.feature_importances_
for i in range(len(importances)):
    feature_dict[feat_labels[i]] = importances[i]

sorted_by_value = sorted(feature_dict.items(), key=lambda kv: kv[1], reverse=True)
print('Feature Rank:')
rank = 1
for k, v in sorted_by_value:
    print(rank, k, v)
    rank += 1

SizeofForest = results.pop('Size of Forest')
results.plot()
plt.title('Random Forest Gini Score Chart')
plt.xlabel('# of Bootstrapes (x10)')
plt.ylabel('% Accuracy')
plt.show()