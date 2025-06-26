
## ⚡ Power Transformer: Box-Cox ও Yeo-Johnson – বাংলায় সহজ ব্যাখ্যা

এই ভিডিওটি আগের ফাংশন ট্রান্সফর্মেশন ভিডিওর পরবর্তী অংশ। এখানে দেখানো হয়েছে কিভাবে **PowerTransformer** ব্যবহার করে ডেটাকে **নরমাল ডিস্ট্রিবিউশন** (Normal Distribution) এর কাছাকাছি আনা যায়, যা অনেক **স্ট্যাটিস্টিকাল ও মেশিন লার্নিং মডেল** এর জন্য দরকারি।

---

### 🔹 PowerTransformer কী?

`PowerTransformer` হলো একটি টেকনিক যা ডেটার **স্কিউনেস (skewness)** কমিয়ে ডেটাকে একটি **সিমেট্রিকাল বা নরমাল শেইপে** আনতে সাহায্য করে। এর মাধ্যমে ফিচারগুলো এমনভাবে রূপান্তরিত হয় যাতে:

* Mean হয় প্রায় **০**
* Standard deviation হয় প্রায় **১**

এতে মডেল ট্রেনিং আরও কার্যকর হয়।

---

## 🧪 ১. Box-Cox Transform

* Box-Cox একটি ম্যাথমেটিক্যাল পদ্ধতি যেটা ডেটার ওপর একটি পাওয়ার ফাংশন প্রয়োগ করে।
* এটা শুধু **positive এবং non-zero** ডেটার ওপর কাজ করে।
* প্রতিটি ফিচারের জন্য একটি **λ (lambda)** ভ্যালু বের করা হয়, যেটা বলে কোন পাওয়ার দিয়ে ট্রান্সফর্ম করলে ডেটা সবচেয়ে বেশি **normally distributed** হয়।
* এই λ নির্ধারণ করার জন্য **MLE (Maximum Likelihood Estimation)** বা **Bayesian estimation** ব্যবহার করা হয়।
* এটি বিশেষভাবে উপকারী যখন আমরা **রিগ্রেশন মডেল** ব্যবহার করি যা ডেটার normality ধরে নেয়।

---

## ⚠️ Box-Cox এর সীমাবদ্ধতা

* এটা **negative অথবা zero value** এর ক্ষেত্রে কাজ করে না।
* তাই ডেটাতে যদি এমন মান থাকে, তাহলে **Box-Cox ব্যর্থ হবে**।

---

## 🧪 ২. Yeo-Johnson Transform

* Yeo-Johnson হলো **Box-Cox এর উন্নত সংস্করণ**।
* এটা **positive, negative এমনকি zero value** সাপোর্ট করে।
* স্বয়ংক্রিয়ভাবে ইনপুটের ধরণ বুঝে নিয়ে মানানসই ট্রান্সফর্মেশন করে।
* তাই **real-world noisy ডেটা**-তে এর কার্যকারিতা অনেক বেশি।

---

## 🔍 বাস্তব উদাহরণ: Concrete Strength Dataset

* ভিডিওতে একটি **Concrete Strength ডেটাসেট** ব্যবহার করা হয়েছে যেটি ছিল অনেক বেশি স্কিউড।
* ট্রান্সফর্ম করার আগে:

  * ডেটাতে **missing values** ও **negative values** আছে কিনা চেক করা হয়।
  * **Linear Regression** প্রয়োগ করে দেখা যায় রেজাল্ট ভালো না (low R² score)।
  * **Histogram** ও **Q-Q plot** দিয়ে দেখা যায় ডেটা ঠিকমতো **normally distributed না**।

---

## 🔁 ট্রান্সফর্মেশন প্রয়োগ

* `sklearn.preprocessing` থেকে **PowerTransformer** ব্যবহার করা হয়েছে।

* প্রথমে Box-Cox প্রয়োগ করে:

  * প্রতিটি ফিচারের জন্য আলাদা λ বের করা হয়।
  * ডেটার distribution কিছুটা উন্নত হয় এবং মডেলের R² scoreও কিছুটা বাড়ে।

* তারপর Yeo-Johnson প্রয়োগ করে:

  * Zero এবং negative values সহ পুরো ডেটাতে কাজ করে।
  * কিছু ক্ষেত্রে Box-Cox এর চেয়ে ভালো রেজাল্ট দেয়।
  * **R² score আরও উন্নত হয়**।

---

## 📊 Before vs After Visualization

* ট্রান্সফর্মেশনের আগে ও পরে বিভিন্ন ফিচারের **distribution graph** দেখানো হয়েছে।
* বিশেষ করে **Fly Ash (ash)** ফিচারটি সবচেয়ে বেশি পরিবর্তিত হয়েছে।
* অনেক ফিচারই transformation এর পর **normal shape** পেয়েছে।

---

## ✅ উপসংহার ও সুপারিশ

* Simple transform যেমন **log, sqrt** ইত্যাদির তুলনায় **PowerTransformer** অনেক বেশি কার্যকর।
* যখন আপনি **linear regression**, **logistic regression** বা অন্য কোন **parametric model** ব্যবহার করবেন, তখন ডেটা **normally distributed** কিনা তা যাচাই করা উচিত।
* যদি না হয়, তাহলে PowerTransformer (Box-Cox বা Yeo-Johnson) প্রয়োগ করা ভালো।
* `sklearn` দিয়ে খুব সহজেই এই ট্রান্সফর্মেশনগুলো কোডে ব্যবহার করা যায়।

---

