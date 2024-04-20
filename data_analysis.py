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
from config import test_configurations


def descriptive_statistics(data_before, data_after):
    desc_stats_before = data_before.describe()
    desc_stats_after = data_after.describe()
    print('Descriptive statistics before utility change:\n', desc_stats_before, '\n')
    print('Descriptive statistics after utility change:\n', desc_stats_after, '\n')


def t_test(data_before, data_after):
    t_stat, p_value = ttest_ind(data_before['Points'], data_after['Points'])
    print(f'T-test for Points: t-statistic = {t_stat}, p-value = {p_value}')
    significance = p_value < 0.05
    return significance


def plot_data(data_before, data_after, label_before, label_after):
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

    plot_average_points(avg_points_before, avg_points_after, label_before, label_after)
    plot_moving_speed(data_before, data_after, label_before, label_after)
    plot_steps(data_before, data_after, label_before, label_after)
    plot_ray_radius(data_before, data_after, label_before, label_after)
    plot_sight(data_before, data_after, label_before, label_after)
    plot_box_weight(data_before, data_after, label_before, label_after)
    plot_boat_weight(data_before, data_after, label_before, label_after)
    plot_volatility(volatility_before, volatility_after, label_before, label_after)

    pass


def regression_analysis(data_before, data_after):
    features = ['Moving Speed', 'Steps', 'Ray Radius', 'Sight', 'Box Weight', 'Weight']
    X_before = data_before[features]
    y_before = data_before['Points']
    X_after = data_after[features]
    y_after = data_after['Points']

    # split the data into training and test sets
    X_train_before, X_test_before, y_train_before, y_test_before = train_test_split(X_before, y_before, test_size=0.2,
                                                                                    random_state=42)
    X_train_after, X_test_after, y_train_after, y_test_after = train_test_split(X_after, y_after, test_size=0.2,
                                                                                random_state=42)
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

    scatter_plot_regression(data_before, data_after, features, regressor_before, regressor_after)
    residual_plot(y_test_before, y_pred_before, y_test_after, y_pred_after)
    coefficient_plot(features, regressor_before, regressor_after)
    prediction_accuracy_plot(y_test_before, y_pred_before, y_test_after, y_pred_after)

    pass


def plot_average_points(avg_points_before, avg_points_after, label_before, label_after):
    plt.figure(figsize=(15, 5))
    bar_width = 0.4

    index = np.arange(len(avg_points_before))

    plt.bar(index, avg_points_before, width=bar_width, color=test_configurations[label_before]['color'], label=test_configurations[label_before]['title'])
    plt.bar(index + bar_width, avg_points_after, width=bar_width, color=test_configurations[label_after]['color'], label=test_configurations[label_after]['title'])

    plt.xlabel('Generations', fontweight='bold')
    plt.ylabel('Average Points', fontweight='bold')
    bin_labels = [f'{int(bin.left)}-{int(bin.right) - 1}' for bin in avg_points_before.index]

    plt.xticks(index + bar_width / 2, bin_labels, rotation=45)
    plt.title(f'Average Points Every Five Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    pass


def plot_moving_speed(data_before, data_after, label_before, label_after):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=data_before, x='Generation', y='Moving Speed', label= test_configurations[label_before]['title'], color=test_configurations[label_before]['color'])
    sns.lineplot(data=data_after, x='Generation', y='Moving Speed', label=test_configurations[label_after]['title'], color=test_configurations[label_after]['color'])
    plt.title(f'Comparison of Moving Speed Over Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.xlabel('Generation')
    plt.ylabel('Moving Speed')
    plt.legend()
    plt.show()
    pass


def plot_steps(data_before, data_after, label_before, label_after):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=data_before, x='Generation', y='Steps', label=test_configurations[label_before]['title'], color=test_configurations[label_before]['color'])
    sns.lineplot(data=data_after, x='Generation', y='Steps', label=test_configurations[label_after]['title'], color=test_configurations[label_after]['color'])
    plt.title(f'Comparison of Steps Over Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.xlabel('Generation')
    plt.ylabel('Steps')
    plt.legend()
    plt.show()

    pass


def plot_ray_radius(data_before, data_after, label_before, label_after):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=data_before, x='Generation', y='Ray Radius', label=test_configurations[label_before]['title'], color=test_configurations[label_before]['color'])
    sns.lineplot(data=data_after, x='Generation', y='Ray Radius', label=test_configurations[label_after]['title'], color=test_configurations[label_after]['color'])
    plt.title(f'Comparison of Ray Radius Over Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.xlabel('Generation')
    plt.ylabel('Ray Radius')
    plt.legend()
    plt.show()
    pass


def plot_sight(data_before, data_after, label_before, label_after):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=data_before, x='Generation', y='Sight', label=test_configurations[label_before]['title'], color=test_configurations[label_before]['color'])
    sns.lineplot(data=data_after, x='Generation', y='Sight', label=test_configurations[label_after]['title'], color=test_configurations[label_after]['color'])
    plt.title(f'Comparison of Sight Over Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.xlabel('Generation')
    plt.ylabel('Sight')
    plt.legend()
    plt.show()
    pass


def plot_box_weight(data_before, data_after, label_before, label_after):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=data_before, x='Generation', y='Box Weight', label=test_configurations[label_before]['title'], color=test_configurations[label_before]['color'])
    sns.lineplot(data=data_after, x='Generation', y='Box Weight', label=test_configurations[label_after]['title'], color=test_configurations[label_after]['color'])
    plt.title(f'Comparison of Box Weight Over Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.xlabel('Generation')
    plt.ylabel('Box Weight')
    plt.legend()
    plt.show()
    pass


def plot_boat_weight(data_before, data_after, label_before, label_after):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=data_before, x='Generation', y='Weight', label=test_configurations[label_before]['title'], color=test_configurations[label_before]['color'])
    sns.lineplot(data=data_after, x='Generation', y='Weight', label=test_configurations[label_after]['title'], color=test_configurations[label_after]['color'])
    plt.title(f'Comparison of Boat Weight Over Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.xlabel('Generation')
    plt.ylabel('Boat Weight')
    plt.legend()
    plt.show()
    pass


def plot_volatility(volatility_before, volatility_after, label_before, label_after):
    bin_size = 5
    plt.figure(figsize=(15, 7))
    plt.plot(volatility_before.index.categories.left + (bin_size / 2), volatility_before,
             label=test_configurations[label_before]['title'], color=test_configurations[label_before]['color'])
    plt.plot(volatility_after.index.categories.left + (bin_size / 2), volatility_after,
             label=test_configurations[label_after]['title'], color=test_configurations[label_after]['color'])
    plt.title(f'Volatility of Points Over Generations: {test_configurations[label_before]["title"]} vs. {test_configurations[label_after]["title"]}')
    plt.xlabel('Generation')
    plt.ylabel('Standard Deviation of Points')
    plt.legend()
    plt.tight_layout()
    plt.show()
    pass


# visual comparison of the regression analysis
# scatter plot with regression lines
def scatter_plot_regression(data_before, data_after, features, regressor_before, regressor_after):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data_before, x=feature, y='Points', color='red', label='Before Change')
        sns.scatterplot(data=data_after, x=feature, y='Points', color='blue', label='After Change')
        plt.title(f'Relationship between {feature} and Points with Regression Line')
        plt.xlabel(feature)
        plt.ylabel('Points')
        plt.legend()
        plt.grid(True)

        sns.regplot(data=data_before, x=feature, y='Points', color='red', scatter=False)
        sns.regplot(data=data_after, x=feature, y='Points', color='blue', scatter=False)
        plt.show()
    pass


def residual_plot(y_test_before, y_pred_before, y_test_after, y_pred_after):
    # calculate residuals
    residuals_before = y_test_before - y_pred_before
    residuals_after = y_test_after - y_pred_after

    # plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_before, residuals_before, color='red', label='Before Change', alpha=0.5)
    plt.scatter(y_pred_after, residuals_after, color='blue', label='After Change', alpha=0.5)
    plt.hlines(y=0, xmin=min(min(y_pred_before), min(y_pred_after)), xmax=max(max(y_pred_before), max(y_pred_after)),
               colors='black', linestyles='--')
    plt.xlabel('Predicted Points')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.show()
    pass


def coefficient_plot(features, regressor_before, regressor_after):
    # coefficient comparison
    coefficients_before = regressor_before.coef_
    coefficients_after = regressor_after.coef_

    indices = np.arange(len(features))
    plt.figure(figsize=(12, 7))
    plt.bar(indices - 0.2, coefficients_before, width=0.4, label='Before Change', color='red')
    plt.bar(indices + 0.2, coefficients_after, width=0.4, label='After Change', color='blue')
    plt.xticks(indices, features, rotation=45)
    plt.ylabel('Coefficient Value')
    plt.title('Comparative Coefficient Plot')
    plt.legend()
    plt.show()
    pass


def prediction_accuracy_plot(y_test_before, y_pred_before, y_test_after, y_pred_after):
    # prediction accuracy plot - to see how well the model predicts the points
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_before, y_pred_before, color='red', label='Before Change', alpha=0.7)
    plt.scatter(y_test_after, y_pred_after, color='blue', label='After Change', alpha=0.7)
    plt.plot([y_test_before.min(), y_test_after.max()], [y_test_before.min(), y_test_after.max()], 'k--', lw=4)
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points')
    plt.title('Actual vs. Predicted Points')
    plt.legend()
    plt.show()
    pass

def load_data(label_before, label_after):
    data_before = pd.read_csv(test_configurations[label_before]['filePathNormal'])
    data_after = pd.read_csv(test_configurations[label_after]['filePathNormal'])
    return data_before, data_after


def main(testType_before, testType_after):
    data_before, data_after = load_data(testType_before, testType_after)
    descriptive_statistics(data_before, data_after)
    significance = t_test(data_before, data_after)
    plot_data(data_before, data_after, testType_before, testType_after)
    regression_analysis(data_before, data_after)
    if significance:
        print('The change in utility function has a statistically significant effect on the points.')
    else:
        print('The change in utility function does not have a statistically significant effect on the points.')
    pass


if __name__ == '__main__':
    main('exponential', 'logarithmic')