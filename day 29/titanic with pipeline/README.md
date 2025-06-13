### Code Cell 1:

```python
import numpy as np
import pandas as pd
```

**ব্যাখ্যা:**

* `numpy` (np): সায়েন্টিফিক ক্যালকুলেশনের জন্য ব্যবহার হয়, যেমন ম্যাট্রিক্স বা অ্যারে অপারেশন।
* `pandas` (pd): ডেটা লোড, প্রসেস, ফিল্টার, ও এনালাইসিস করার জন্য সবচেয়ে জনপ্রিয় লাইব্রেরি।

---

###  Code Cell 2:

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
```

**ব্যাখ্যা:**

* `train_test_split`: ডেটাকে ট্রেনিং ও টেস্ট অংশে ভাগ করে।
* `ColumnTransformer`: বিভিন্ন কলামে ভিন্ন ভিন্ন ট্রান্সফরমেশন (যেমন স্কেল, এনকোড) প্রয়োগ করার জন্য।
* `SimpleImputer`: ডেটাতে যেসব ভ্যালু মিসিং, সেগুলা পূরণ করে (যেমন গড় দিয়ে বা মোড দিয়ে)।
* `OneHotEncoder`: ক্যাটাগরিকাল ডেটাকে সংখ্যায় রূপান্তর করে।
* `MinMaxScaler`: সব ভ্যালুকে 0-1 রেঞ্জে স্কেল করে।
* `Pipeline/make_pipeline`: পুরো প্রিপ্রসেসিং ধাপগুলোকে একটা পাইপলাইনে সাজায়, যাতে একসাথে ব্যবহার করা যায়।
* `SelectKBest, chi2`: সবচেয়ে গুরুত্বপূর্ণ Kটি ফিচার নির্বাচন করে (chi-squared টেস্ট ব্যবহার করে)।
* `DecisionTreeClassifier`: ডিসিশন ট্রি মডেল, যেটা ক্লাসিফিকেশন প্রোবলেম সলভ করতে ব্যবহৃত হয়।

---

###  Code Cell 3:

```python
df = pd.read_csv('train.csv')
```

**ব্যাখ্যা:**

* `pd.read_csv()`: `'train.csv'` নামের ডেটাসেট ফাইলটা লোড করে `df` নামক একটা pandas ডেটাফ্রেমে রাখে।



###  Code Cell 4:

```python
df.head()
```

**ব্যাখ্যা:**

* এটা `train.csv` ফাইল থেকে লোড করা `df` ডেটাফ্রেমের প্রথম ৫টি রো (সারি) দেখায়।
* মূলত ডেটা দেখতে কেমন সেটা প্রাথমিকভাবে চেক করার জন্য ব্যবহৃত হয়।

---

###  Code Cell 5:

```python
df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
```

**ব্যাখ্যা:**

* এই লাইনে কিছু অপ্রয়োজনীয় কলাম (যেমন আইডি, নাম, টিকিট নম্বর, ক্যাবিন) ডেটা থেকে ড্রপ করে দিচ্ছে।
* `inplace=True` মানে ডেটাফ্রেমে সরাসরি এই পরিবর্তনটা হয়ে যাচ্ছে, আলাদা করে কিছু রিটার্ন নিচ্ছে না।

---

###  Code Cell 6:

```python
# Step 1 -> train/test/split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Survived']),
                                                    df['Survived'],
                                                    test_size=0.2,
                                                    random_state=42)
```

**ব্যাখ্যা:**

* এখানে ডেটা ট্রেনিং আর টেস্ট সেটে ভাগ করা হচ্ছে।
* `X_train` ও `X_test` → ফিচার/ইনপুট ডেটা।
* `y_train` ও `y_test` → টার্গেট লেবেল (মানে যাকে প্রেডিক্ট করবো, এখানে 'Survived')।
* `test_size=0.2` → ২০% ডেটা টেস্ট করার জন্য রাখা হয়েছে।
* `random_state=42` → যাতে রেজাল্ট বারবার একই হয় (রিপ্রোডিউসিবিলিটির জন্য)।

---

###  Code Cell 7:

```python
X_train.head()
```

**ব্যাখ্যা:**

* ট্রেনিং ডেটার ইনপুট অংশ (X\_train) এর প্রথম ৫টি রো প্রিন্ট করে দেখানো হয়েছে।
* মূলত যাচাই করার জন্য যে ডেটা সঠিকভাবে ভাগ হয়েছে কিনা।



###  Code Cell 8:

```python
y_train.sample(5)
```

**ব্যাখ্যা:**

* `y_train` (target labels) থেকে যেকোনো ৫টা র‍্যান্ডম স্যাম্পল দেখানো হয়েছে।
* এতে দেখা যায় কারা বেঁচে গেছে (`1`) আর কারা মারা গেছে (`0`)।

---

###  Code Cell 9:

```python
# imputation transformer
trf1 = ColumnTransformer([
    ('impute_age', SimpleImputer(), [2]),
    ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])
], remainder='passthrough')
```

**ব্যাখ্যা:**

* এখানে মিসিং ডেটা পূরণ করার জন্য একটি `ColumnTransformer` তৈরি করা হয়েছে।
* `SimpleImputer()` → যেখানে `NaN` আছে, সেখানে ডিফল্টভাবে mean বা অন্য value বসিয়ে দেয়।
* `impute_age` → ৩ নম্বর কলাম (সম্ভবত Age) এ মিসিং ডেটা পূরণ করবে।
* `impute_embarked` → ৭ নম্বর কলামে সবচেয়ে বেশি ব্যবহৃত ভ্যালু (mode) বসিয়ে দেবে।
* `remainder='passthrough'` → বাকি কলামগুলো অপরিবর্তিত থাকবে।

---

###  Code Cell 10:

```python
# one hot encoding
trf2 = ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(sparse=False, handle_unknown='ignore'), [1, 6])
], remainder='passthrough')
```

**ব্যাখ্যা:**

* এখানে ক্যাটাগরিকাল ডেটাগুলাকে নাম্বারে রূপান্তর করার জন্য `OneHotEncoder` ব্যবহার করা হয়েছে।
* `OneHotEncoder` → ক্যাটাগরি অনুযায়ী আলাদা কলাম বানিয়ে দেয় (যেমন Male, Female → দুটি কলাম)।
* `[1, 6]` → সম্ভবত Sex এবং Embarked কলামগুলা এনকোড করা হচ্ছে।
* `sparse=False` → আউটপুট হিসেবে array রিটার্ন করবে, sparse matrix না।
* `handle_unknown='ignore'` → যদি ট্রেনিংয়ে না থাকা নতুন ক্যাটাগরি দেখা যায়, সেটাকে ইগনোর করবে।

---

###  Code Cell 11:

```python
# Scaling
trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0, 10))
])
```

**ব্যাখ্যা:**

* `MinMaxScaler()` → সব ভ্যালু 0 থেকে 1 এর মধ্যে নিয়ে আসে, যাতে মডেল ভালো ট্রেইন হয়।
* `slice(0, 10)` → প্রথম ১০টি কলাম স্কেল করা হবে।

