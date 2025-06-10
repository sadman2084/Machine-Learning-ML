

### Column Transformer - Realme Wise Explanation

1. **কি?**
   Column Transformer হলো একটা স্কিকিট-লার্ন (sklearn) এর টুল যা ডেটাসেটের বিভিন্ন কলামে ভিন্ন ভিন্ন প্রক্রিয়া (transformation) প্রয়োগ করার জন্য।

2. **কেন?**
   ডেটায় বিভিন্ন ধরনের ডেটা থাকে — যেমন: সংখ্যাসূচক (numerical) এবং ক্যাটেগোরিক্যাল (categorical)। এগুলোতে আলাদা আলাদা প্রক্রিয়া দরকার। একসাথে সব ডেটাতে একই প্রক্রিয়া দিলে ভুল হয়।

3. **কিভাবে?**
   Column Transformer ব্যবহার করে, তুমি সহজেই বলতে পারো কোন কলামে কি প্রক্রিয়া চালাতে হবে। তারপর সেটা সব একসাথে করিয়ে নেয়া যায়।

4. **কী কী হয়?**

   * Numerical কলামে যেমন StandardScaler, MinMaxScaler ব্যবহার করতে পারো।
   * Categorical কলামে যেমন OneHotEncoder, OrdinalEncoder ব্যবহার করতে পারো।

5. **কোড উদাহরণ**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'salary']),
        ('cat', OneHotEncoder(), ['gender', 'city'])
    ])
```

6. **ব্যবহার**
   এরপর `preprocessor.fit_transform(X)` দিলে, `X` এর `age` আর `salary` কলামগুলো স্কেল হবে, আর `gender` আর `city` ওয়ান-হট এনকোডিং হবে।

7. **ফায়দা**

   * কোড পরিষ্কার ও সঠিক হয়
   * একবারে সব প্রক্রিয়া করা যায়
   * কম্পাইল বা পাইপলাইনে ব্যবহার করা সহজ হয়

---

### ছোট্ট সারমর্ম:

**Column Transformer = একসাথে আলাদা আলাদা কলামে আলাদা আলাদা প্রক্রিয়া চালানোর টুল।**

---


