## Example 1: MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[100, 200], [150, 300], [200, 400]])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)
```
**Explanation:**
- **MinMaxScaler** scales data to a range between 0 and 1 by subtracting the minimum value and dividing by the range (max - min).
- This transformation is useful for ensuring that data values are on a similar scale.

---

## Example 2: StandardScaler
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[100, 200], [150, 300], [200, 400]])

scaler = StandardScaler()
standard_data = scaler.fit_transform(data)

print(standard_data)
```
**Explanation:**
- **StandardScaler** standardizes the data by removing the mean and scaling to unit variance. It is often used when features have different units or scales.
- The formula used is: `(x - mean) / standard deviation`.

---

## Example 3: RobustScaler
```python
from sklearn.preprocessing import RobustScaler
import numpy as np

data = np.array([[100, 200], [150, 300], [200, 400]])

scaler = RobustScaler()
robust_data = scaler.fit_transform(data)

print(robust_data)
```
**Explanation:**
- **RobustScaler** uses the median and interquartile range for scaling, which makes it robust to outliers in the data.
- This method is useful when data contains many outliers.

---

## Example 4: T-test
```python
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import chi2_contingency

g1 = [20, 33, 21, 53, 23]
g2 = [32, 52, 62, 63, 64]

t_stat, p_value = stats.ttest_ind(g1, g2)

print(f'T_statistic: {t_stat}')
print(f'p_value: {p_value}')
```
**Explanation:**
- A **T-test** is used to compare the means of two independent groups (here `g1` and `g2`) to see if they are significantly different from each other.
- `t_stat` indicates the test statistic, and `p_value` helps to determine whether the results are statistically significant.

---

## Example 5: Chi-Square Test
```python
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import chi2_contingency

data = pd.DataFrame({
    'g1': [20, 30],
    'g2': [25, 35]
}, index=['category 1', 'category 2'])

chi2_stat, p_value, _, _ = chi2_contingency(data)

print(f'T_statistic: {chi2_stat}')
print(f'p_value: {p_value}')
```
**Explanation:**
- The **Chi-square test** checks whether there is a significant association between categorical variables. It compares observed frequencies with expected frequencies.
- `chi2_stat` is the test statistic, and `p_value` is the probability that the result is due to chance.

---

## Example 6: LabelEncoder
```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = ['low', 'medium', 'high', 'medium', 'low']

encoder = LabelEncoder()

encoder_data = encoder.fit_transform(data)
print(encoder_data)
```
**Explanation:**
- **LabelEncoder** converts categorical labels (like 'low', 'medium', 'high') into numeric values.
- It assigns an integer to each unique label, making it useful for machine learning models that require numerical input.

---

## Example 7: OrdinalEncoder
```python
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

df = pd.DataFrame({'size': ['Small', 'Medium', 'Large', 'Medium', 'Large']})

encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])

encoder_data = encoder.fit_transform(df[['size']])
print(encoder_data)
```
**Explanation:**
- **OrdinalEncoder** encodes categories with a meaningful order (e.g., 'Small', 'Medium', 'Large').
- The `categories` parameter defines the order of the categories, which is preserved during encoding.

---

## Example 8: One-Hot Encoding
```python
import pandas as pd

df = pd.DataFrame({'color': ['red', 'green', 'blue', 'green', 'red']})

one_hot_encoded_data = pd.get_dummies(df, columns=['color'])

print(one_hot_encoded_data)
```
**Explanation:**
- **One-Hot Encoding** converts categorical variables into a format that can be provided to ML algorithms to improve predictions.
- Each category is represented as a binary column, where 1 represents the presence of that category and 0 represents its absence.

---

## Example 9: Random Over Sampling
```python
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import numpy as np
import pandas as pd

x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 0, 0, 1])

ros = RandomOverSampler(random_state=42)

x_resampled, y_resampled = ros.fit_resample(x, y)

print(f"Resampled y: {Counter(y_resampled)}")
```
**Explanation:**
- **RandomOverSampler** increases the number of minority class samples by randomly duplicating them.
- This is commonly used to handle class imbalance in classification tasks, ensuring that each class has enough representation.

---

## Example 10: Random Over Sampling with SMOTE
```python
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
import numpy as np

x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 0, 0, 1])

ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(x, y)
print(f"Resampled y with RandomOverSampler: {Counter(y_ros)}")

smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x_ros, y_ros)
print(f"Resampled y with SMOTE: {Counter(y_smote)}")
```
**Explanation:**
- **SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic samples for the minority class rather than duplicating existing ones.
- It is used to improve the performance of classification algorithms on imbalanced datasets.

---

## Example 11: Linear Regression
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')

plt.title("Linear Regression")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```
**Explanation:**
- **Linear Regression** fits a linear model to the data, aiming to find the line that best predicts `y` from `x`.
- In this example, the relationship is quadratic (y = x^2), but the linear regression tries to fit a straight line.

---

## Example 12: Logistic Regression
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
```
**Explanation:**
- **Logistic Regression** is used for binary classification tasks. It estimates probabilities using a logistic function.
- The `classification_report` provides performance metrics like precision, recall, and F1-score.

---

## Example 13: Date Range in Pandas
```python
import pandas as pd
import numpy as np

dates = pd.date_range(start='2024-08-19', periods=10, freq='D')

data = pd.DataFrame({'value': np.random.randint(1, 100, size=(10,))}, index=dates)

print(data)
```
**Explanation:**
- **pd.date_range** generates a sequence of dates starting from a specific date and with a defined frequency.
- This is useful for creating time series data.

---

## Example 14: Time Series Plot
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range(start='2024-08-19', periods=10, freq='D')

data = pd.DataFrame({'value': np.random.randint(1, 100, size=(10,))}, index=dates)

data.plot()
plt.title("Time Series Data")
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```
**Explanation:**
- This example plots time series data, using `matplotlib` to visualize the values over time.

---

## Example 15: Seasonal Decomposition of Time Series
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
data = pd.Series(
    10 + 0.1 * np.arange(100) + np.sin(2 * np.pi * np.arange(100) / 12),
    index=dates
)

decomposition = seasonal_decompose(data, model='additive')
decomposition.plot()
plt.show()
```
**Explanation:**
- **Seasonal decomposition** breaks down time series data into trend, seasonal, and residual components.
- This helps in understanding the underlying patterns and fluctuations in data.

---

## Example 16: ARIMA Forecasting
```python
import pandas as

 pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
data = pd.Series(
    10 + 0.1 * np.arange(100) + np.sin(2 * np.pi * np.arange(100) / 12),
    index=dates
)

model = ARIMA(data, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=10)

plt.plot(data.index, data, label='Original')
plt.plot(pd.date_range(start=data.index[-1], periods=11, freq='D')[1:], forecast, label='Forecast', color='red')
plt.legend()
plt.show()
```
**Explanation:**
- **ARIMA (AutoRegressive Integrated Moving Average)** is used for forecasting time series data.
- It uses past data points and trends to predict future values.
