import pandas as pd
# Resolution settings
w = 800
h = 600

# Seed State
set_seed = 0

# Default data frame settings
df = pd.read_csv(r'C:\Users\Gordon\PycharmProjects\GUI Project\Data\iris.csv', header=None)
df_data = df.iloc[:, 0:4]
df_target = df.iloc[:, 4]

summ_status = 0
freq_status = 0
scatter_status = 0

# Classification
testsize = 0.25
user_n = 3