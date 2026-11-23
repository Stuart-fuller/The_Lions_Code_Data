import pandas as pd;
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score;

df = pd.read_csv("data/fingers_crossed.csv");

X = df.drop(columns=["final_result", "region"]);
y = df["final_result"].replace(2, 1);


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
);

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
);


n_arr = [];
accuracy_arr = [];
precision_arr = [];
recall_arr = [];
f1_arr = [];

for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i);
    knn.fit(X_train, y_train);
    
    val_predictions = knn.predict(X_val);
    test_predictions = knn.predict(X_test);
    
    acc = accuracy_score(y_test, test_predictions);
    prec = precision_score(y_test, test_predictions, average="weighted");
    rec = recall_score(y_test, test_predictions, average="weighted");
    f1 = f1_score(y_test, test_predictions, average="weighted");
    
    accuracy_arr.append(acc);
    precision_arr.append(prec);
    recall_arr.append(rec);
    f1_arr.append(f1);
    n_arr.append(i);
    continue;

plt.plot(n_arr, accuracy_arr);
plt.plot(n_arr, precision_arr);
plt.plot(n_arr, recall_arr);
plt.plot(n_arr, f1_arr);

plt.xlabel("Cluster Count");
plt.ylabel("Metric")

plt.legend(["Accuracy", "Precision", "Recall", "F1"]);

plt.show();