import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\LENOVO\Desktop\canada_per_capita_income.csv")
df.head()

plt.xlabel('year')
plt.ylabel('capital')
plt.scatter(df.year,df.capital,color='red',marker='+')
plt.show()
new_df = df.drop('capital',axis='columns')
capital = df.capital

reg = linear_model.LinearRegression()
reg.fit(new_df,capital)
print(reg.predict([[2020]]))