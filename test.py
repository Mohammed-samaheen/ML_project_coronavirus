import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# loud the data and make some preprocess
data = pd.read_csv('covid_19_clean_complete.csv')
re_data = data.groupby('Date').sum()

data["Date"] = data["Date"].apply(lambda x: pd.Timestamp(x))
re_data = data.groupby('Date', as_index=False).sum()
re_data.head()

# plot showing the rate of Confirmed,Deaths and Recovered  of virus corona in Spain
fig, axes = plt.subplots(figsize=(20, 5))
axes.plot(re_data["Date"], re_data["Confirmed"], label="Confirmed", marker='o')
axes.plot(re_data["Date"], re_data["Deaths"], label="Deaths")
axes.plot(re_data["Date"], re_data["Recovered"], label="Recovered")
axes.legend()
plt.xticks(re_data["Date"][::2])
plt.gcf().autofmt_xdate()
plt.show()
