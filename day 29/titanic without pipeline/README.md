
### ✅ **\[Code Cell 1] লাইব্রেরি ইমপোর্ট:**

```python
import numpy as np
import pandas as pd
```

* `numpy` (np): সংখ্যাগত গণনার জন্য ব্যবহৃত হয়, বিশেষত ম্যাট্রিক্স ও অ্যারে।
* `pandas` (pd): ডেটা প্রসেসিং ও টেবিল আকারে ডেটা (DataFrame) ম্যানিপুলেশনের জন্য।

```python
from sklearn.model_selection import train_test_split
```

* `train_test_split`: ডেটাসেটকে ট্রেনিং ও টেস্ট অংশে ভাগ করে। Random ভাবে ভাগ হয়।

```python
from sklearn.impute import SimpleImputer
```

* `SimpleImputer`: Missing (null) value গুলো পূরণ করতে ব্যবহৃত হয় (যেমন mean, median বা mode দিয়ে)।

```python
from sklearn.preprocessing import OneHotEncoder
```

* `OneHotEncoder`: ক্যাটেগরিকাল ভ্যালুগুলোকে 0 ও 1 আকারে রূপান্তর করে (একটি কলাম = একটি ক্যাটেগরি)।

```python
from sklearn.preprocessing import MinMaxScaler
```

* `MinMaxScaler`: ডেটাকে একটি নির্দিষ্ট স্কেলে (সাধারণত 0 থেকে 1) মানে রূপান্তর করে। যদিও এখানে এটা ব্যবহৃত হয়নি।

```python
from sklearn.tree import DecisionTreeClassifier
```

* `DecisionTreeClassifier`: একটি ক্লাসিফিকেশন মডেল যেটা ডিসিশন ট্রি ভিত্তিক।

---

### ✅ **\[Code Cell 2] CSV ডেটা লোড:**

```python
df = pd.read_csv('train.csv')
```

* `pd.read_csv()`: `.csv` ফাইল থেকে ডেটা পড়ে `DataFrame` বানায়।
* এখানে `df` হচ্ছে মূল ডেটাফ্রেম যাতে Titanic ডেটাসেট লোড হয়েছে।

---

### ✅ **\[Code Cell 3] ডেটা প্রিভিউ:**

```python
df.head()
```

* `head()`: DataFrame এর প্রথম ৫টি সারি দেখায়।

---

### ✅ **\[Code Cell 4] অপ্রয়োজনীয় কলাম বাদ:**

```python
df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
```

* `drop()`: নির্দিষ্ট কলাম বাদ দেয়।
* `inplace=True`: স্থায়ীভাবে `df` থেকে কলামগুলো বাদ দেয়।

---

### ✅ **\[Code Cell 5] আবার প্রিভিউ:**

```python
df.head()
```

* পূর্বের মতোই, পরিবর্তনের পর ডেটা কেমন হয়েছে তা দেখতে।

---

### ✅ **\[Code Cell 6] ডেটা ভাগ করা (Train/Test Split):**

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['Survived']),
    df['Survived'],
    test_size=0.2,
    random_state=42
)
```

* `X_train`: প্রশিক্ষণের জন্য ইনপুট।
* `X_test`: পরীক্ষার জন্য ইনপুট।
* `y_train`: প্রশিক্ষণের জন্য আউটপুট (Survived)।
* `y_test`: পরীক্ষার জন্য আউটপুট।
* `test_size=0.2`: ২০% ডেটা টেস্টে, ৮০% ট্রেনিংয়ে।
* `random_state=42`: একই ভাগ বারবার পেতে fixed seed।

---

### ✅ **\[Code Cell 7-8] ইনপুট ও আউটপুট দেখানো:**

```python
X_train.head(2)
y_train.head()
```

* প্রথম ২টি সারি ও লক্ষ্য (target) ডেটা দেখা হয়েছে।

---

### ✅ **\[Code Cell 9] Missing Value Count:**

```python
df.isnull().sum()
```

* `isnull()`: কোন কোন ঘরে null আছে তা জানায়।
* `sum()`: প্রতিটি কলামে কতগুলো missing আছে, তা হিসেব করে।

---

### ✅ **\[Code Cell 10] Missing Value Imputation:**

```python
si_age = SimpleImputer()
si_embarked = SimpleImputer(strategy='most_frequent')
```

* `si_age`: ডিফল্টভাবে mean দিয়ে পূরণ করবে।
* `si_embarked`: সবচেয়ে ঘনঘটা ভ্যালু (mode) দিয়ে পূরণ করবে।

```python
X_train_age = si_age.fit_transform(X_train[['Age']])
X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])

X_test_age = si_age.transform(X_test[['Age']])
X_test_embarked = si_embarked.transform(X_test[['Embarked']])
```

* `fit_transform()`: ট্রেন ডেটায় ফিট করে ও মান পূরণ করে।
* `transform()`: আগে শিখে রাখা মান দিয়ে টেস্ট ডেটায় Impute করে।

---

### ✅ **\[Code Cell 11] One Hot Encoding:**

```python
ohe_sex = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_embarked = OneHotEncoder(sparse=False, handle_unknown='ignore')
```

* `sparse=False`: আউটপুট numpy array হবে।
* `handle_unknown='ignore'`: নতুন ক্যাটেগরি এলে ignore করবে।

```python
X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)

X_test_sex = ohe_sex.transform(X_test[['Sex']])
X_test_embarked = ohe_embarked.transform(X_test_embarked)
```

* `fit_transform()`: প্রথমে ট্রেন ডেটায় ফিট করে ও রূপান্তর করে।
* `transform()`: টেস্ট ডেটায় একই রূপান্তর প্রয়োগ।

---

### ✅ **\[Code Cell 12-13] রিমেইনিং ডেটা রাখা:**

```python
X_train_rem = X_train.drop(columns=['Sex','Age','Embarked'])
X_test_rem = X_test.drop(columns=['Sex','Age','Embarked'])
```

* পূর্বের ডেটা থেকে যেগুলো encode বা impute হয়েছে সেগুলো বাদ দিয়ে বাকি ফিচার রাখা হয়েছে।

---

### ✅ **\[Code Cell 14] সব ফিচার একত্রে জোড়া:**

```python
X_train_transformed = np.concatenate((X_train_rem, X_train_age, X_train_sex, X_train_embarked), axis=1)
X_test_transformed = np.concatenate((X_test_rem, X_test_age, X_test_sex, X_test_embarked), axis=1)
```

* `np.concatenate`: সব ফিচার একত্রে numpy array বানায় (axis=1 মানে কলাম হিসেবে যুক্ত করা)।

---

### ✅ **\[Code Cell 15-16] Decision Tree Train এবং Predict:**

```python
clf = DecisionTreeClassifier()
clf.fit(X_train_transformed, y_train)
```

* Decision tree তৈরি করা ও ট্রেনিং।

```python
y_pred = clf.predict(X_test_transformed)
```

* টেস্ট ডেটায় প্রেডিকশন।

---

### ✅ **\[Code Cell 17] Accuracy Check:**

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

* প্রেডিকশন কতটা সঠিক হয়েছে তা মাপা হয় accuracy দিয়ে।

---

### ✅ **\[Code Cell 18] মডেল ও এনকোডার সংরক্ষণ:**

```python
import pickle

pickle.dump(ohe_sex, open('models/ohe_sex.pkl', 'wb'))
pickle.dump(ohe_embarked, open('models/ohe_embarked.pkl', 'wb'))
pickle.dump(clf, open('models/clf.pkl', 'wb'))
```

* `pickle.dump(obj, file, 'wb')`: নির্দিষ্ট অবজেক্ট ফাইল হিসেবে সংরক্ষণ করে future use এর জন্য।

