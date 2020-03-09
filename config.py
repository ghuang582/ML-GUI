import pandas as pd
# Resolution settings
w = 1000
h = 600

# Seed State
set_seed = 0

# Default data frame settings
df = pd.read_csv(r'C:\Users\Gordon\PycharmProjects\GUI Project\Data\iris.csv', header=None)
data_cols = 4
target_col = 4
df_data = df.iloc[:, :data_cols]
df_target = df.iloc[:, target_col]

summ_status = 0
freq_status = 0
scatter_status = 0

selected = "Nearest Neighbours"
# Classification
## Nearest Neighbours
kn_test_size = 0.25
user_n = 3