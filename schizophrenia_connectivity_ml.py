import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

np.random.seed(42)

# =====================================================
# 1. GENERACIÓN DE DATOS (controlando el ruido)
# =====================================================

def generate_typical(n=20, noise_level=0.15):
    mat = np.random.normal(0.5, noise_level, (n,n))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1)
    return mat

def generate_schizophrenia(n=20, noise_level=0.15, extra_noise=0.08):
    mat = np.random.normal(0.5, noise_level, (n,n))
    
    # perturbación global
    mat += np.random.normal(0, extra_noise, (n,n))
    
    # perturbación local (desorganización)
    mask = np.random.rand(n,n) < 0.3
    mat[mask] += np.random.normal(0, 0.1, np.sum(mask))
    
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1)
    return mat

# =====================================================
# 2. FEATURES
# =====================================================

def features_basic(mat):
    return [
        np.mean(mat),
        np.std(mat)
    ]

def features_extended(mat):
    local_block = mat[:5, :5]
    return [
        np.mean(mat),
        np.std(mat),
        np.mean(local_block) - np.mean(mat)
    ]

# =====================================================
# 3. EXPERIMENTO
# =====================================================

def run_experiment(feature_type="basic", noise_level=0.15):

    n_samples = 200
    data, labels = [], []

    for _ in range(n_samples):
        data.append(generate_typical(noise_level=noise_level))
        labels.append(0)

    for _ in range(n_samples):
        data.append(generate_schizophrenia(noise_level=noise_level))
        labels.append(1)

    if feature_type == "basic":
        X = np.array([features_basic(m) for m in data])
    else:
        X = np.array([features_extended(m) for m in data])

    y = np.array(labels)

    # Normalización
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo (uno solo para análisis limpio)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv = cross_val_score(model, X, y, cv=5).mean()

    return acc, cv, model, X_test, y_test


# =====================================================
# 4. COMPARACIÓN ENTRE FEATURES
# =====================================================

noise_levels = [0.10, 0.15, 0.20, 0.25]

results_basic = []
results_extended = []

print("\n=== EXPERIMENTOS ===")

for noise in noise_levels:
    acc_b, cv_b, _, _, _ = run_experiment("basic", noise)
    acc_e, cv_e, _, _, _ = run_experiment("extended", noise)

    results_basic.append(cv_b)
    results_extended.append(cv_e)

    print(f"\nNoise level: {noise}")
    print(f"Basic (Mean+Std): CV = {cv_b:.3f}")
    print(f"Extended (+Local): CV = {cv_e:.3f}")

# =====================================================
# 5. GRÁFICO (esto es oro para tu paper)
# =====================================================

plt.plot(noise_levels, results_basic, marker='o', label="Mean + Std")
plt.plot(noise_levels, results_extended, marker='o', label="Mean + Std + Local")

plt.xlabel("Noise Level")
plt.ylabel("Cross-Validation Accuracy")
plt.title("Feature Robustness to Noise")
plt.legend()
plt.grid()
plt.show()

# =====================================================
# 6. FEATURE IMPORTANCE (modelo final)
# =====================================================

acc, cv, model, X_test, y_test = run_experiment("extended", 0.15)

print("\n=== FEATURE IMPORTANCE ===")

feature_names = ["Mean", "Std", "Local vs Global"]

r = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)

for i in range(len(feature_names)):
    print(f"{feature_names[i]}: {r.importances_mean[i]:.4f}")


####BOXPLOT Y T-TEST
from scipy.stats import ttest_ind

# =====================================================
# FUNCIÓN QUE DEVUELVE SCORES (NO SOLO PROMEDIO)
# =====================================================

def get_cv_scores(feature_type="basic", noise_level=0.15):

    n_samples = 200
    data, labels = [], []

    for _ in range(n_samples):
        data.append(generate_typical(noise_level=noise_level))
        labels.append(0)

    for _ in range(n_samples):
        data.append(generate_schizophrenia(noise_level=noise_level))
        labels.append(1)

    if feature_type == "basic":
        X = np.array([features_basic(m) for m in data])
    else:
        X = np.array([features_extended(m) for m in data])

    y = np.array(labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, X, y, cv=5)

    return scores

# =====================================================
# 7. BOXPLOT
# =====================================================
# Elegimos un nivel de ruido (el más interesante: 0.15)
noise = 0.15

scores_basic = get_cv_scores("basic", noise)
scores_extended = get_cv_scores("extended", noise)

# Boxplot
plt.boxplot([scores_basic, scores_extended], tick_labels=["Basic", "Extended"])
plt.title(f"Model Comparison (noise={noise})")
plt.ylabel("Accuracy")
plt.grid()
plt.show()



# =====================================================
# 8. T-TEST
# =====================================================
t_stat, p_value = ttest_ind(scores_basic, scores_extended)

print("\n=== T-TEST ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")



# =====================================================
# 9. GRÁFICA PARA EL ABSTRACT
# =====================================================
plt.figure(figsize=(6,4))

plt.plot(noise_levels, results_basic, marker='o', linewidth=2,  linestyle='-', label="Mean + Std")
plt.plot(noise_levels, results_extended, marker='o', linewidth=2, linestyle='--', label="Mean + Std + Local")

plt.xlabel("Noise level")
plt.ylabel("Cross-validation accuracy")
plt.title("Effect of noise on classification performance")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()