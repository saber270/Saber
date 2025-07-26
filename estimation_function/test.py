# 优化标准logistic从0.96提升到0.97的各种技巧

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv('detailed_generation_speeds.csv')
X = df['Batch Size'].values.reshape(-1, 1)
y = df['Generation Speed'].values

def logistic_standard(x, L, k, x0):
    """标准logistic函数"""
    return L / (1 + np.exp(-k * (x - x0)))

print("=== 优化标准logistic：从0.96到0.97 ===")

# 基准结果（我们知道的0.96）
print("\n基准结果:")
try:
    popt_base, pcov_base = curve_fit(logistic_standard, X.flatten(), y, 
                                    p0=[80, 0.03, 10], maxfev=5000)
    y_pred_base = logistic_standard(X.flatten(), *popt_base)
    r2_base = r2_score(y, y_pred_base)
    print(f"基准配置: R² = {r2_base:.4f}")
except:
    r2_base = 0.96
    print(f"基准配置: R² ≈ 0.96 (估算)")

# === 优化技巧1: 全局优化算法 ===
print("\n1. 全局优化算法 (Differential Evolution):")

def objective_function(params):
    """目标函数：最小化MSE"""
    try:
        L, k, x0 = params
        y_pred = logistic_standard(X.flatten(), L, k, x0)
        mse = mean_squared_error(y, y_pred)
        return mse
    except:
        return 1e10

# 设置参数边界
bounds = [
    (50, 200),      # L的范围
    (0.001, 0.5),   # k的范围  
    (1, 60)         # x0的范围
]

try:
    result = differential_evolution(objective_function, bounds, seed=42, maxiter=1000)
    popt_global = result.x
    y_pred_global = logistic_standard(X.flatten(), *popt_global)
    r2_global = r2_score(y, y_pred_global)
    print(f"  全局优化: R² = {r2_global:.4f}")
    print(f"  参数: L={popt_global[0]:.2f}, k={popt_global[1]:.4f}, x0={popt_global[2]:.2f}")
    
    if r2_global >= 0.969:
        print("  ✅ 达到≥0.97目标!")
except Exception as e:
    print(f"  全局优化失败: {e}")

# === 优化技巧2: 数据预处理 ===
print("\n2. 数据预处理优化:")

# 2a. 异常值处理
def remove_outliers_iqr(X, y, factor=1.5):
    """使用IQR方法移除异常值"""
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    return X[mask], y[mask]

try:
    X_clean, y_clean = remove_outliers_iqr(X.flatten(), y)
    popt_clean, pcov_clean = curve_fit(logistic_standard, X_clean, y_clean, 
                                      p0=[80, 0.03, 10], maxfev=5000)
    # 在原始数据上评估
    y_pred_clean = logistic_standard(X.flatten(), *popt_clean)
    r2_clean = r2_score(y, y_pred_clean)
    print(f"  去异常值: R² = {r2_clean:.4f} (移除{len(y)-len(y_clean)}个点)")
    
    if r2_clean >= 0.969:
        print("  ✅ 达到≥0.97目标!")
except Exception as e:
    print(f"  去异常值失败: {e}")

# 2b. 数据平滑
from scipy.ndimage import gaussian_filter1d

try:
    # 对y进行轻微平滑
    sort_idx = np.argsort(X.flatten())
    y_sorted = y[sort_idx]
    y_smoothed = gaussian_filter1d(y_sorted, sigma=1.0)
    y_smooth_original_order = np.zeros_like(y)
    y_smooth_original_order[sort_idx] = y_smoothed
    
    popt_smooth, pcov_smooth = curve_fit(logistic_standard, X.flatten(), y_smooth_original_order, 
                                        p0=[80, 0.03, 10], maxfev=5000)
    # 在原始数据上评估
    y_pred_smooth = logistic_standard(X.flatten(), *popt_smooth)
    r2_smooth = r2_score(y, y_pred_smooth)
    print(f"  数据平滑: R² = {r2_smooth:.4f}")
    
    if r2_smooth >= 0.969:
        print("  ✅ 达到≥0.97目标!")
except Exception as e:
    print(f"  数据平滑失败: {e}")

# === 优化技巧3: 更好的初始参数估计 ===
print("\n3. 智能初始参数估计:")

def estimate_logistic_params(X, y):
    """智能估计logistic参数"""
    x_flat = X.flatten()
    
    # 排序数据
    sort_idx = np.argsort(x_flat)
    x_sorted = x_flat[sort_idx]
    y_sorted = y[sort_idx]
    
    # 估计L：使用最大值加一点余量
    L_est = np.max(y) * 1.05
    
    # 估计x0：找到y值接近中值的位置
    y_mid = (np.max(y) + np.min(y)) / 2
    idx_mid = np.argmin(np.abs(y_sorted - y_mid))
    x0_est = x_sorted[idx_mid]
    
    # 估计k：基于斜率
    # 找到20%-80%的位置计算斜率
    y_20 = np.percentile(y, 20)
    y_80 = np.percentile(y, 80)
    idx_20 = np.argmin(np.abs(y_sorted - y_20))
    idx_80 = np.argmin(np.abs(y_sorted - y_80))
    
    if idx_80 > idx_20:
        slope = (y_sorted[idx_80] - y_sorted[idx_20]) / (x_sorted[idx_80] - x_sorted[idx_20])
        k_est = abs(slope) / (L_est / 4)  # logistic斜率的近似关系
    else:
        k_est = 0.05
    
    return [L_est, k_est, x0_est]

try:
    smart_initial = estimate_logistic_params(X, y)
    popt_smart, pcov_smart = curve_fit(logistic_standard, X.flatten(), y, 
                                      p0=smart_initial, maxfev=10000)
    y_pred_smart = logistic_standard(X.flatten(), *popt_smart)
    r2_smart = r2_score(y, y_pred_smart)
    print(f"  智能初始参数: R² = {r2_smart:.4f}")
    print(f"  初始估计: L={smart_initial[0]:.2f}, k={smart_initial[1]:.4f}, x0={smart_initial[2]:.2f}")
    print(f"  最终参数: L={popt_smart[0]:.2f}, k={popt_smart[1]:.4f}, x0={popt_smart[2]:.2f}")
    
    if r2_smart >= 0.969:
        print("  ✅ 达到≥0.97目标!")
except Exception as e:
    print(f"  智能初始参数失败: {e}")

# === 优化技巧4: 多重随机初始化 ===
print("\n4. 多重随机初始化:")

best_r2 = 0
best_params = None
n_trials = 50

np.random.seed(42)
for i in range(n_trials):
    # 随机生成初始参数
    L_init = np.random.uniform(60, 150)
    k_init = np.random.uniform(0.01, 0.2)
    x0_init = np.random.uniform(5, 40)
    
    try:
        popt_random, pcov_random = curve_fit(logistic_standard, X.flatten(), y, 
                                           p0=[L_init, k_init, x0_init], maxfev=5000)
        y_pred_random = logistic_standard(X.flatten(), *popt_random)
        r2_random = r2_score(y, y_pred_random)
        
        if r2_random > best_r2:
            best_r2 = r2_random
            best_params = popt_random
    except:
        continue

print(f"  多重初始化最佳: R² = {best_r2:.4f}")
if best_params is not None:
    print(f"  最佳参数: L={best_params[0]:.2f}, k={best_params[1]:.4f}, x0={best_params[2]:.2f}")

if best_r2 >= 0.969:
    print("  ✅ 达到≥0.97目标!")

# === 优化技巧5: 优化算法参数调整 ===
print("\n5. 优化算法参数调整:")

optimization_configs = [
    ({'method': 'lm', 'maxfev': 10000}, "Levenberg-Marquardt"),
    ({'method': 'trf', 'maxfev': 10000, 'ftol': 1e-15, 'xtol': 1e-15}, "Trust Region + 高精度"),
    ({'method': 'dogbox', 'maxfev': 10000}, "Dogbox"),
]

for config, name in optimization_configs:
    try:
        popt_opt, pcov_opt = curve_fit(logistic_standard, X.flatten(), y, 
                                      p0=[80, 0.03, 10], **config)
        y_pred_opt = logistic_standard(X.flatten(), *popt_opt)
        r2_opt = r2_score(y, y_pred_opt)
        print(f"  {name:20s}: R² = {r2_opt:.4f}")
        
        if r2_opt >= 0.969:
            print("    ✅ 达到≥0.97目标!")
    except Exception as e:
        print(f"  {name:20s}: 失败")

# === 总结最佳方法 ===
print("\n" + "="*50)
print("🎯 总结:")
print("你的同事可能使用了以下优化技巧之一:")
print("1. 全局优化算法 (differential_evolution)")
print("2. 更精确的优化算法参数")
print("3. 智能的初始参数估计")
print("4. 多重随机初始化选择最佳结果")
print("5. 轻微的数据预处理")

print(f"\n如果找到了R²≥0.97的方法，那就是答案!")
print("这些都是合理的优化技巧，不涉及修改模型本身。")