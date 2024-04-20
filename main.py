import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

filePathBefore = 'C:\\Users\\patgh\\Documents\\SaxionCMGT\\Year 2\\Term 3\\Advanced Tools\\Data From Smart Boats\\pirateBoatsData250Generations.csv'
filePathAfter = 'C:\\Users\\patgh\\Documents\\SaxionCMGT\\Year 2\\Term 3\\Advanced Tools\\Data From Smart Boats\\pirateBoatsData250GenerationsQuadraticUtil.csv'

data_before = pd.read_csv(filePathBefore)
data_after = pd.read_csv(filePathAfter)

# calculate descriptive statistics
desc_stats_before = data_before.describe()
desc_stats_after = data_after.describe()

#display descriptive statistics
print('Descriptive statistics before utility change:\n', desc_stats_before, '\n')
print('Descriptive statistics after utility change:\n', desc_stats_after, '\n')

#t-test for statistical difference in Points
t_stat, p_value = ttest_ind(data_before['Points'], data_after['Points'])
print(f'T-test for Points: t-statistic = {t_stat}, p-value = {p_value}')

#check if the difference in Points is statistically significant
if p_value < 0.05:
    print('The change in utility function has a statistically significant effect on the points.')
else:
    print('The change in utility function does not have a statistically significant effect on the points.')

#plot the average points every five generations
bin_size = 5
max_generation = min(data_before['Generation'].max(), data_after['Generation'].max())
generation_bins = range(0, max_generation + bin_size, bin_size)

data_before['GenerationBin'] = pd.cut(data_before['Generation'], bins=generation_bins)
data_after['GenerationBin'] = pd.cut(data_after['Generation'], bins=generation_bins)

avg_points_before = data_before.groupby('GenerationBin')['Points'].mean()
avg_points_after = data_after.groupby('GenerationBin')['Points'].mean()

# calculate volatility
volatility_before = data_before.groupby('GenerationBin')['Points'].std()
volatility_after = data_after.groupby('GenerationBin')['Points'].std()

# regression analysis
features = ['Moving Speed', 'Steps', 'Ray Radius', 'Sight', 'Box Weight', 'Weight']
X_before = data_before[features]
y_before = data_before['Points']
X_after = data_after[features]
y_after = data_after['Points']

# split the data into training and test sets
X_train_before, X_test_before, y_train_before, y_test_before = train_test_split(X_before, y_before, test_size=0.2, random_state=42)
X_train_after, X_test_after, y_train_after, y_test_after = train_test_split(X_after, y_after, test_size=0.2, random_state=42)

# train the linear regression model
regressor_before = LinearRegression()
regressor_after = LinearRegression()

# fit the model
regressor_before.fit(X_train_before, y_train_before)
regressor_after.fit(X_train_after, y_train_after)

# make predictions
y_pred_before = regressor_before.predict(X_test_before)
y_pred_after = regressor_after.predict(X_test_after)

# model evaluation
mse_before = mean_squared_error(y_test_before, y_pred_before)
mse_after = mean_squared_error(y_test_after, y_pred_after)
r2_before = r2_score(y_test_before, y_pred_before)
r2_after = r2_score(y_test_after, y_pred_after)

print(f'Before Utility change - MSE: {mse_before}, R^2: {r2_before}')
print(f'After Utility change - MSE: {mse_after}, R^2: {r2_after}')


plt.figure(figsize=(15, 5))
bar_width = 0.4

index = np.arange(len(avg_points_before))

plt.bar(index, avg_points_before, width=bar_width, color='red', label='Linear Utility Function')
plt.bar(index + bar_width, avg_points_after, width=bar_width, color='blue', label='Quadratic Utility Function')

plt.xlabel('Generations', fontweight='bold')
plt.ylabel('Average Points', fontweight='bold')
bin_labels = [f'{int(bin.left)}-{int(bin.right)-1}' for bin in avg_points_before.index]

plt.xticks(index + bar_width / 2, bin_labels, rotation=45)
plt.title('Average Points Every Five Generations')
plt.legend()
plt.tight_layout()
plt.show()


#visual comparison of of other variables, in this case I am looking at the how the change in utility function affects the sight of the boats
#visual comparison of the moving speed
sns.set_style('whitegrid')
plt.figure(figsize=(15, 5))
sns.lineplot(data=data_before, x='Generation', y='Moving Speed', label='Linear Utility Function', color='red')
sns.lineplot(data=data_after, x='Generation', y='Moving Speed', label='Quadratic Utility Function', color='blue')
plt.title('Comparison of Moving Speed Over Generations')
plt.xlabel('Generation')
plt.ylabel('Moving Speed')
plt.legend()
plt.show()

#visual comparison of the Steps of the boats
sns.set_style('whitegrid')
plt.figure(figsize=(15, 5))
sns.lineplot(data=data_before, x='Generation', y='Steps', label='Linear Utility Function', color='red')
sns.lineplot(data=data_after, x='Generation', y='Steps', label='Quadratic Utility Function', color='blue')
plt.title('Comparison of Steps Over Generations')
plt.xlabel('Generation')
plt.ylabel('Steps')
plt.legend()
plt.show()

# visual comparison of the ray radius
sns.set_style('whitegrid')
plt.figure(figsize=(15, 5))
sns.lineplot(data=data_before, x='Generation', y='Ray Radius', label='Linear Utility Function', color='red')
sns.lineplot(data=data_after, x='Generation', y='Ray Radius', label='Quadratic Utility Function', color='blue')
plt.title('Comparison of Ray Radius Over Generations')
plt.xlabel('Generation')
plt.ylabel('Ray Radius')
plt.legend()
plt.show()

# visual comparison of the Sight
sns.set_style('whitegrid')
plt.figure(figsize=(15, 5))
sns.lineplot(data=data_before, x='Generation', y='Sight', label='Linear Utility Function', color='red')
sns.lineplot(data=data_after, x='Generation', y='Sight', label='Quadratic Utility Function', color='blue')
plt.title('Comparison of Sight Over Generations')
plt.xlabel('Generation')
plt.ylabel('Sight')
plt.legend()
plt.show()

# visual comparison of the Box Weight
sns.set_style('whitegrid')
plt.figure(figsize=(15, 5))
sns.lineplot(data=data_before, x='Generation', y='Box Weight', label='Linear Utility Function', color='red')
sns.lineplot(data=data_after, x='Generation', y='Box Weight', label='Quadratic Utility Function', color='blue')
plt.title('Comparison of Box Weight Over Generations')
plt.xlabel('Generation')
plt.ylabel('Box Weight')
plt.legend()

# visual comparison of the Boat Weight
sns.set_style('whitegrid')
plt.figure(figsize=(15, 5))
sns.lineplot(data=data_before, x='Generation', y='Weight', label='Linear Utility Function', color='red')
sns.lineplot(data=data_after, x='Generation', y='Weight', label='Quadratic Utility Function', color='blue')
plt.title('Comparison of Boat Weight Over Generations')
plt.xlabel('Generation')
plt.ylabel('Boat Weight')
plt.legend()
plt.show()


