import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit


# Load data from CSV
df = pd.read_csv('measured_generation_speed.csv')

# Prepare data
X = df['Batch Size'].values.reshape(-1, 1)
y = df['Generation Speed'].values

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)

slope = linear_model.coef_[0]
intercept = linear_model.intercept_

print("Linear Regression Results:")
print(f"Linear model: speed = {intercept:.2f} + {slope:.4f} * load")
print(f"MSE: {mse_linear:.4f}, R²: {r2_linear:.4f}")

# Logistic Function Fitting
def logistic_function(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

initial_guess = [np.max(y), 0.01, np.median(X.flatten())]

try:
    popt_logistic, pcov = curve_fit(logistic_function, X.flatten(), y, 
                                   p0=initial_guess, maxfev=10000)
    
    y_pred_logistic = logistic_function(X.flatten(), *popt_logistic)
    mse_logistic = mean_squared_error(y, y_pred_logistic)
    r2_logistic = r2_score(y, y_pred_logistic)
    
    print("\nLogistic Function Results:")
    print(f"Logistic model: speed = {popt_logistic[0]:.2f} / (1 + exp(-{popt_logistic[1]:.4f} * (load - {popt_logistic[2]:.2f})))")
    print(f"MSE: {mse_logistic:.4f}, R²: {r2_logistic:.4f}")
    
except Exception as e:
    print(f"\nLogistic fitting failed: {e}")
    mse_logistic = float('inf')
    r2_logistic = -1

# USL Model
def usl_model(x, a, sigma, kappa):
    denominator = 1 + sigma * x + kappa * x**2
    return a / denominator

try:
    popt_usl, _ = curve_fit(usl_model, X.flatten(), y, 
                           p0=[y[0], 0.1, 0.01], maxfev=10000)
    
    y_pred_usl = usl_model(X.flatten(), *popt_usl)
    mse_usl = mean_squared_error(y, y_pred_usl)
    r2_usl = r2_score(y, y_pred_usl)
    
    print("\nUSL Model Results:")
    print(f"USL model: speed = {popt_usl[0]:.2f} / (1 + {popt_usl[1]:.4f} * load + {popt_usl[2]:.4f} * load^2)")
    print(f"MSE: {mse_usl:.4f}, R²: {r2_usl:.4f}")
    
except RuntimeError as e:
    print(f"\nUSL model fitting failed: {e}")
    mse_usl = float('inf')
    r2_usl = -1

# Comprehensive summary and save to file
summary_lines = []
summary_lines.append("Model Comparison Results")
summary_lines.append("=" * 50)
summary_lines.append("")

# Linear regression details
summary_lines.append("Linear Regression Results:")
summary_lines.append(f"Linear model: speed = {intercept:.2f} + {slope:.4f} * load")
summary_lines.append(f"MSE: {mse_linear:.4f}, R²: {r2_linear:.4f}")
summary_lines.append("")

# Logistic function details
if mse_logistic < float('inf'):
    summary_lines.append("Logistic Function Results:")
    summary_lines.append(f"Logistic model: speed = {popt_logistic[0]:.2f} / (1 + exp(-{popt_logistic[1]:.4f} * (load - {popt_logistic[2]:.2f})))")
    summary_lines.append(f"MSE: {mse_logistic:.4f}, R²: {r2_logistic:.4f}")
    summary_lines.append("")

# USL model details
if mse_usl < float('inf'):
    summary_lines.append("USL Model Results:")
    summary_lines.append(f"USL model: speed = {popt_usl[0]:.2f} / (1 + {popt_usl[1]:.4f} * load + {popt_usl[2]:.4f} * load^2)")
    summary_lines.append(f"MSE: {mse_usl:.4f}, R²: {r2_usl:.4f}")
    summary_lines.append("")

# Summary table
summary_lines.append("Summary Table:")
summary_lines.append(f"{'Model':<12} {'MSE':<12} {'R²':<12}")
summary_lines.append("-" * 36)
summary_lines.append(f"{'Linear':<12} {mse_linear:<12.4f} {r2_linear:<12.4f}")
if mse_logistic < float('inf'):
    summary_lines.append(f"{'Logistic':<12} {mse_logistic:<12.4f} {r2_logistic:<12.4f}")
if mse_usl < float('inf'):
    summary_lines.append(f"{'USL':<12} {mse_usl:<12.4f} {r2_usl:<12.4f}")

# Print to console
for line in summary_lines:
    print(line)

# Save to file
with open('model_comparison_summary.txt', 'w') as f:
    for line in summary_lines:
        f.write(line + '\n')

print("\nDetailed summary saved to 'model_comparison_summary.txt'")