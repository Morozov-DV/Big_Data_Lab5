# -*- coding: utf-8 -*-
"""
Лабораторна робота №5. Варіант 14.
Тема: розв'язання задач класифікації та регресії із застосуванням одиночних
та ансамблевих моделей машинного навчання.
Індивідуальне завдання (варіант 14): дослідити перенавчання дерева регресії
залежно від глибини та мінімального розміру листа.

Власний набір даних: «Енергоспоживання житлових будинків» (15000 записів).
  - регресія:      energy_kwh  (місячне споживання електроенергії, кВт·год)
  - класифікація:  over_limit  (1 - споживання перевищує нормативний ліміт)
"""
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # збереження графіків у файли
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     cross_val_score, GridSearchCV,
                                     validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              BaggingClassifier, BaggingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              VotingClassifier, StackingClassifier, StackingRegressor)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110
FIG = "figs/"
RES = {}                                    # підсумкові числа для звіту


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def save(name):
    plt.tight_layout()
    plt.savefig(FIG + name, dpi=150)
    plt.close()


# ======================================================================
# 1. ГЕНЕРАЦІЯ ВЛАСНОГО НАБОРУ ДАНИХ
# ======================================================================
np.random.seed(14)                          # відтворюваність (номер варіанта)
n = 15000                                   # обсяг у межах 10000-20000

area = np.clip(np.random.lognormal(4.3, 0.35, n), 25, 400)        # площа, м2
floors = np.random.randint(1, 10, n)                              # поверховість
building_age = np.clip(np.random.exponential(30, n), 0, 100)      # вік будинку
insulation = np.clip(np.random.normal(6, 2, n), 1, 10)            # утеплення 1-10
window_ratio = np.random.uniform(0.1, 0.5, n)                     # частка вікон
heating_temp = np.random.normal(2, 6, n)                          # сер. t зовн., °C
residents = np.clip(np.random.poisson(2.2, n) + 1, 1, 8)          # мешканці
appliances = np.random.randint(3, 16, n)                          # прилади
noise_1 = np.random.normal(0, 1, n)                               # шумові ознаки
noise_2 = np.random.normal(0, 1, n)
noise_3 = np.random.uniform(-1, 1, n)

# Регресійна ціль: нелінійні складники + взаємодія площі та утеплення
energy_kwh = (
    80
    + 2.0 * area + 0.006 * area ** 2                 # квадратична залежність
    + 0.0035 * building_age ** 2                     # нелінійний вплив віку
    + 0.12 * area * (10 - insulation)                # взаємодія ознак
    + 25 * (insulation < 3)                          # порогова залежність
    - 6.0 * heating_temp + 0.15 * heating_temp ** 2  # нелінійність за температурою
    + 18 * residents + 4 * appliances
    + 40 * window_ratio
    + np.random.normal(0, 15, n)                     # випадковий шум (sigma = 15)
)

# Класифікаційна ціль: ймовірність перевищення ліміту (логістична модель)
logit = (
    -2.6
    + 0.50 * (6 - insulation)
    + 0.025 * (building_age - 30)
    + 0.012 * (area - 75)
    + 0.18 * (2 - heating_temp) / 2
    + 0.7 * ((residents >= 5) & (appliances > 10))   # взаємодія
)
prob = 1 / (1 + np.exp(-logit))
over_limit = np.random.binomial(1, prob)

df = pd.DataFrame({
    "area": area, "floors": floors, "building_age": building_age,
    "insulation": insulation, "window_ratio": window_ratio,
    "heating_temp": heating_temp, "residents": residents,
    "appliances": appliances, "noise_1": noise_1, "noise_2": noise_2,
    "noise_3": noise_3, "energy_kwh": energy_kwh, "over_limit": over_limit})

print(df.head()); print(df.shape); print(df.describe().round(2).T)

# ======================================================================
# 2. ПЕРВИННИЙ АНАЛІЗ ДАНИХ
# ======================================================================
print(df.info())
print("Пропущені значення:\n", df.isna().sum().sum())
print("Частки класів:\n", df["over_limit"].value_counts(normalize=True).round(3))
RES["n"] = n
RES["share_class1"] = float(df["over_limit"].mean())
RES["mean_energy"] = float(df["energy_kwh"].mean())
RES["std_energy"] = float(df["energy_kwh"].std())

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
cnt = df["over_limit"].value_counts().sort_index()
ax[0].bar(["У межах ліміту (0)", "Перевищення (1)"], cnt.values, color=["#4C78A8", "#E45756"])
ax[0].set_title("Розподіл класів цільової змінної over_limit")
ax[0].set_ylabel("Кількість будинків"); ax[0].grid(True, axis="y", alpha=.4)
ax[1].hist(df["energy_kwh"], bins=40, color="#59A14F", edgecolor="white")
ax[1].set_title("Розподіл цільової змінної energy_kwh")
ax[1].set_xlabel("Споживання, кВт·год"); ax[1].set_ylabel("Кількість будинків")
ax[1].grid(True, alpha=.4)
save("fig1_distributions.png")

corr = df.corr()
plt.figure(figsize=(10, 8.5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
plt.title("Кореляційна матриця ознак і цільових змінних")
save("fig2_corr.png")

# ======================================================================
# 3. РОЗДІЛЕННЯ НА НАВЧАЛЬНУ ТА ТЕСТОВУ ВИБІРКИ (stratify за класом)
# ======================================================================
features = ["area", "floors", "building_age", "insulation", "window_ratio",
            "heating_temp", "residents", "appliances",
            "noise_1", "noise_2", "noise_3"]
X, y_cls, y_reg = df[features], df["over_limit"], df["energy_kwh"]
X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=14, stratify=y_cls)
print("Навчальна:", X_train.shape, "Тестова:", X_test.shape)
print("Частка класу 1:", round(yc_train.mean(), 3), round(yc_test.mean(), 3))
RES["n_train"], RES["n_test"] = len(X_train), len(X_test)
RES["share_train"], RES["share_test"] = float(yc_train.mean()), float(yc_test.mean())

# ======================================================================
# 4. КЛАСИФІКАЦІЯ: ОДИНОЧНІ ТА АНСАМБЛЕВІ МОДЕЛІ
# ======================================================================
results_cls, fitted_cls = [], {}


def evaluate_classifier(name, model):
    t0 = time.time()
    model.fit(X_train, yc_train)
    fit_t = time.time() - t0
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    row = {"Модель": name,
           "Accuracy навч.": model.score(X_train, yc_train),
           "Accuracy тест": accuracy_score(yc_test, pred),
           "Precision": precision_score(yc_test, pred),
           "Recall": recall_score(yc_test, pred),
           "F1": f1_score(yc_test, pred),
           "ROC-AUC": roc_auc_score(yc_test, proba),
           "Час навч., с": fit_t}
    results_cls.append(row); fitted_cls[name] = model
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


log_reg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
knn_c = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=25))])
tree_c_full = DecisionTreeClassifier(random_state=14)
tree_c_5 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=14)
forest_c = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=14)
bag_c = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8, min_samples_leaf=10),
                          n_estimators=100, n_jobs=-1, random_state=14)
ada_c = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                           n_estimators=200, learning_rate=0.5, random_state=14)
gb_c = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                  subsample=0.8, random_state=14)
hgb_c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=14)
voting_c = VotingClassifier([("lr", log_reg), ("knn", knn_c), ("tree", tree_c_5)], voting="soft")
stack_c = StackingClassifier([("lr", log_reg), ("forest", forest_c), ("gb", gb_c)],
                             final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1)

for nm, mdl in [("Логістична регресія", log_reg), ("k найближчих сусідів", knn_c),
                ("Дерево без обмежень", tree_c_full), ("Дерево обмежене", tree_c_5),
                ("Випадковий ліс", forest_c), ("Бегінг дерев", bag_c), ("AdaBoost", ada_c),
                ("Градієнтний бустинг", gb_c), ("Гістограмний бустинг", hgb_c),
                ("Голосування", voting_c), ("Стекінг", stack_c)]:
    evaluate_classifier(nm, mdl)

table_cls = pd.DataFrame(results_cls).set_index("Модель")
table_cls.round(4).to_csv("table_classification.csv")
print(table_cls.round(4))

# залежність точності класифікаційного дерева від глибини
depths_c = range(1, 21)
tr_c, te_c = [], []
for d in depths_c:
    t = DecisionTreeClassifier(max_depth=d, random_state=14).fit(X_train, yc_train)
    tr_c.append(t.score(X_train, yc_train)); te_c.append(t.score(X_test, yc_test))
RES["cls_best_depth"] = int(list(depths_c)[int(np.argmax(te_c))])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
ax[0].plot(depths_c, tr_c, "o-", label="Навчальна вибірка")
ax[0].plot(depths_c, te_c, "s-", label="Тестова вибірка")
ax[0].set_xlabel("Максимальна глибина дерева"); ax[0].set_ylabel("Accuracy")
ax[0].set_title("Класифікація: якість дерева залежно від глибини")
ax[0].legend(); ax[0].grid(True)
order = table_cls.sort_values("ROC-AUC").index
ax[1].barh(order, table_cls.loc[order, "ROC-AUC"], color="#F58518")
ax[1].set_xlim(0.6, table_cls["ROC-AUC"].max() + 0.02)
ax[1].set_title("ROC-AUC моделей класифікації (тест)"); ax[1].grid(True, axis="x", alpha=.4)
save("fig3_cls_depth_and_auc.png")

# матриці неточностей
best_name_c = table_cls["ROC-AUC"].idxmax()
RES["cls_best_name"] = best_name_c
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, nm in zip(axes, ["Логістична регресія", best_name_c]):
    ConfusionMatrixDisplay.from_estimator(fitted_cls[nm], X_test, yc_test,
        display_labels=["У межах", "Перевищення"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(nm)
save("fig4_confusion.png")

# ROC-криві
plt.figure(figsize=(8, 6.5))
for nm, mdl in fitted_cls.items():
    fpr, tpr, _ = roc_curve(yc_test, mdl.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=nm)
plt.plot([0, 1], [0, 1], "k--", label="Випадкове вгадування")
plt.xlabel("Частка хибнопозитивних"); plt.ylabel("Частка істинно позитивних")
plt.title("ROC-криві моделей класифікації"); plt.legend(fontsize=8); plt.grid(True)
save("fig5_roc.png")

# ======================================================================
# 5. РЕГРЕСІЯ: ОДИНОЧНІ ТА АНСАМБЛЕВІ МОДЕЛІ
# ======================================================================
results_reg, fitted_reg = [], {}


def evaluate_regressor(name, model):
    t0 = time.time()
    model.fit(X_train, yr_train)
    fit_t = time.time() - t0
    pred = model.predict(X_test)
    row = {"Модель": name,
           "R2 навч.": model.score(X_train, yr_train),
           "R2 тест": r2_score(yr_test, pred),
           "MAE": mean_absolute_error(yr_test, pred),
           "RMSE": rmse(yr_test, pred),
           "Час навч., с": fit_t}
    results_reg.append(row); fitted_reg[name] = model
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
tree_r_full = DecisionTreeRegressor(random_state=14)
tree_r = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=14)
knn_r = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=15))])
forest_r = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=14)
bag_r = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=10, min_samples_leaf=10),
                         n_estimators=100, n_jobs=-1, random_state=14)
ada_r = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4),
                          n_estimators=200, learning_rate=0.5, random_state=14)
gb_r = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, random_state=14)
hgb_r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=14)
stack_r = StackingRegressor([("lin", lin), ("forest", forest_r), ("gb", gb_r)],
                            final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)

for nm, mdl in [("Лінійна регресія", lin), ("Ridge-регресія", ridge),
                ("Дерево без обмежень", tree_r_full), ("Дерево обмежене", tree_r),
                ("k найближчих сусідів", knn_r), ("Випадковий ліс", forest_r),
                ("Бегінг дерев", bag_r), ("AdaBoost", ada_r),
                ("Градієнтний бустинг", gb_r), ("Гістограмний бустинг", hgb_r),
                ("Стекінг", stack_r)]:
    evaluate_regressor(nm, mdl)

table_reg = pd.DataFrame(results_reg).set_index("Модель")
table_reg.round(4).to_csv("table_regression.csv")
print(table_reg.round(4))
best_name_r = table_reg["RMSE"].idxmin()
RES["reg_best_name"] = best_name_r

order = table_reg.sort_values("RMSE", ascending=False).index
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax[0].barh(order, table_reg.loc[order, "RMSE"], color="#E45756")
ax[0].set_title("RMSE на тестовій вибірці"); ax[0].grid(True, axis="x", alpha=.4)
ax[1].barh(order, table_reg.loc[order, "R2 тест"], color="#59A14F")
ax[1].set_xlim(0.8, 1.0); ax[1].set_title("R² на тестовій вибірці")
ax[1].grid(True, axis="x", alpha=.4)
save("fig6_reg_compare.png")

# фактичні та прогнозовані значення
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for a, nm in zip(ax, ["Лінійна регресія", best_name_r]):
    p = fitted_reg[nm].predict(X_test)
    a.scatter(yr_test, p, s=8, alpha=.4)
    lo, hi = yr_test.min(), yr_test.max()
    a.plot([lo, hi], [lo, hi], "r--", label="Ідеальна відповідність")
    a.set_title(nm); a.set_xlabel("Фактичні значення, кВт·год")
    a.set_ylabel("Прогнозовані значення, кВт·год"); a.legend(); a.grid(True)
save("fig7_pred_vs_actual.png")

# аналіз залишків
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for i, nm in enumerate(["Лінійна регресія", best_name_r]):
    p = fitted_reg[nm].predict(X_test); r = yr_test - p
    ax[i, 0].hist(r, bins=40, color="#4C78A8", edgecolor="white")
    ax[i, 0].set_title(f"{nm}. Розподіл залишків"); ax[i, 0].set_xlabel("Залишок"); ax[i, 0].grid(True)
    ax[i, 1].scatter(p, r, s=8, alpha=.4); ax[i, 1].axhline(0, color="r", ls="--")
    ax[i, 1].set_title(f"{nm}. Залишки відносно прогнозу")
    ax[i, 1].set_xlabel("Прогноз"); ax[i, 1].set_ylabel("Залишок"); ax[i, 1].grid(True)
save("fig8_residuals.png")

# ======================================================================
# 6. ІНДИВІДУАЛЬНЕ ЗАВДАННЯ (ВАРІАНТ 14):
#    перенавчання дерева регресії залежно від глибини та мінімального
#    розміру листа
# ======================================================================
def fit_tree(depth, leaf):
    return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                                 random_state=14).fit(X_train, yr_train)


def tree_metrics(t):
    ptr, pte = t.predict(X_train), t.predict(X_test)
    return {"rmse_tr": rmse(yr_train, ptr), "rmse_te": rmse(yr_test, pte),
            "r2_tr": r2_score(yr_train, ptr), "r2_te": r2_score(yr_test, pte),
            "leaves": int(t.get_n_leaves()), "depth_real": int(t.get_depth())}


NOISE = 15.0                                   # sigma шуму в даних => межа RMSE
RES["noise_floor"] = NOISE

# --- 6.1 Залежність від глибини (min_samples_leaf = 1) -----------------
depths = list(range(1, 26))
A = {leaf: [tree_metrics(fit_tree(d, leaf)) for d in depths] for leaf in [1, 5, 20, 50]}

# 5-кратна крос-валідація по глибині (leaf = 1) - стійка оцінка
cv5 = KFold(n_splits=5, shuffle=True, random_state=14)
tr_sc, va_sc = validation_curve(
    DecisionTreeRegressor(random_state=14), X_train, yr_train,
    param_name="max_depth", param_range=depths, cv=cv5,
    scoring="neg_root_mean_squared_error", n_jobs=-1)
cv_rmse_depth = -va_sc.mean(axis=1); cv_rmse_depth_std = va_sc.std(axis=1)
best_d_cv = depths[int(np.argmin(cv_rmse_depth))]
best_d_test = depths[int(np.argmin([m["rmse_te"] for m in A[1]]))]
RES["depth_best_cv"] = best_d_cv
RES["depth_best_test"] = best_d_test
RES["depth_cv_rmse_at_best"] = float(cv_rmse_depth.min())

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(depths, [m["rmse_tr"] for m in A[1]], "o-", label="Навчальна вибірка")
ax[0].plot(depths, [m["rmse_te"] for m in A[1]], "s-", label="Тестова вибірка")
ax[0].errorbar(depths, cv_rmse_depth, yerr=cv_rmse_depth_std, fmt="^--", capsize=2,
               alpha=.7, label="Крос-валідація (5 фолдів)")
ax[0].axhline(NOISE, color="gray", ls=":", label=f"Межа шуму ≈ {NOISE:.0f}")
ax[0].axvline(best_d_cv, color="green", ls="--", alpha=.6)
ax[0].set_xlabel("Максимальна глибина дерева"); ax[0].set_ylabel("RMSE, кВт·год")
ax[0].set_title("Дерево регресії: RMSE залежно від глибини (min_samples_leaf = 1)")
ax[0].legend(); ax[0].grid(True)
for leaf, st in zip([1, 5, 20, 50], ["-", "--", "-.", ":"]):
    ax[1].plot(depths, [m["rmse_te"] for m in A[leaf]], st, marker="o", ms=3,
               label=f"min_samples_leaf = {leaf}")
ax[1].axhline(NOISE, color="gray", ls=":")
ax[1].set_xlabel("Максимальна глибина дерева"); ax[1].set_ylabel("Тестовий RMSE, кВт·год")
ax[1].set_title("Вплив мінімального розміру листа на тестову помилку")
ax[1].legend(); ax[1].grid(True)
save("fig9_depth_study.png")

# складність дерева (кількість листів) та розрив між навчанням і тестом
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].semilogy(depths, [m["leaves"] for m in A[1]], "o-")
ax[0].set_xlabel("Максимальна глибина дерева"); ax[0].set_ylabel("Кількість листів (лог. шкала)")
ax[0].set_title("Зростання складності дерева"); ax[0].grid(True)
ax[1].plot(depths, [m["r2_tr"] - m["r2_te"] for m in A[1]], "o-", color="#E45756")
ax[1].set_xlabel("Максимальна глибина дерева"); ax[1].set_ylabel("R² навч. − R² тест")
ax[1].set_title("Розрив якості (показник перенавчання)"); ax[1].grid(True)
save("fig10_complexity_gap.png")

# таблиця режимів для leaf = 1
rows = []
for d in [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25]:
    m = A[1][depths.index(d)]
    rows.append({"Глибина": d, "Листів": m["leaves"],
                 "RMSE навч.": m["rmse_tr"], "RMSE тест": m["rmse_te"],
                 "R2 навч.": m["r2_tr"], "R2 тест": m["r2_te"],
                 "Розрив R2": m["r2_tr"] - m["r2_te"]})
table_depth = pd.DataFrame(rows); table_depth.round(4).to_csv("table_depth.csv", index=False)
print(table_depth.round(4))

# --- 6.2 Залежність від мінімального розміру листа (без обмеження глибини)
leaves = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400]
B = [tree_metrics(fit_tree(None, l)) for l in leaves]
tr_l, va_l = validation_curve(
    DecisionTreeRegressor(random_state=14), X_train, yr_train,
    param_name="min_samples_leaf", param_range=leaves, cv=cv5,
    scoring="neg_root_mean_squared_error", n_jobs=-1)
cv_rmse_leaf = -va_l.mean(axis=1); cv_rmse_leaf_std = va_l.std(axis=1)
best_l_cv = leaves[int(np.argmin(cv_rmse_leaf))]
RES["leaf_best_cv"] = best_l_cv
RES["leaf_cv_rmse_at_best"] = float(cv_rmse_leaf.min())

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].semilogx(leaves, [m["rmse_tr"] for m in B], "o-", label="Навчальна вибірка")
ax[0].semilogx(leaves, [m["rmse_te"] for m in B], "s-", label="Тестова вибірка")
ax[0].errorbar(leaves, cv_rmse_leaf, yerr=cv_rmse_leaf_std, fmt="^--", capsize=2,
               alpha=.7, label="Крос-валідація (5 фолдів)")
ax[0].axhline(NOISE, color="gray", ls=":", label=f"Межа шуму ≈ {NOISE:.0f}")
ax[0].set_xlabel("Мінімальний розмір листа (лог. шкала)"); ax[0].set_ylabel("RMSE, кВт·год")
ax[0].set_title("RMSE залежно від min_samples_leaf (глибина не обмежена)")
ax[0].legend(); ax[0].grid(True)
ax[1].semilogx(leaves, [m["r2_tr"] for m in B], "o-", label="R² навчальна")
ax[1].semilogx(leaves, [m["r2_te"] for m in B], "s-", label="R² тестова")
ax[1].set_xlabel("Мінімальний розмір листа (лог. шкала)"); ax[1].set_ylabel("R²")
ax[1].set_title("R² залежно від min_samples_leaf"); ax[1].legend(); ax[1].grid(True)
save("fig11_leaf_study.png")

table_leaf = pd.DataFrame([{"min_samples_leaf": l, "Листів": m["leaves"],
                            "Реальна глибина": m["depth_real"],
                            "RMSE навч.": m["rmse_tr"], "RMSE тест": m["rmse_te"],
                            "R2 навч.": m["r2_tr"], "R2 тест": m["r2_te"]}
                           for l, m in zip(leaves, B)])
table_leaf.round(4).to_csv("table_leaf.csv", index=False)
print(table_leaf.round(4))

# --- 6.3 Двовимірна сітка: глибина × мінімальний розмір листа ----------
g_depths = [2, 4, 6, 8, 10, 12, 15, 20, None]
g_leaves = [1, 3, 5, 10, 20, 50, 100, 200]
Z_te = np.zeros((len(g_leaves), len(g_depths)))
Z_gap = np.zeros_like(Z_te)
for i, l in enumerate(g_leaves):
    for j, d in enumerate(g_depths):
        m = tree_metrics(fit_tree(d, l))
        Z_te[i, j] = m["rmse_te"]; Z_gap[i, j] = m["rmse_te"] - m["rmse_tr"]
xt = [str(d) if d else "∞" for d in g_depths]

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for a, Z, ttl, cm in [(ax[0], Z_te, "Тестовий RMSE", "viridis_r"),
                      (ax[1], Z_gap, "Розрив: RMSE тест − RMSE навч.", "magma_r")]:
    im = a.imshow(Z, cmap=cm, aspect="auto"); plt.colorbar(im, ax=a)
    a.set_xticks(range(len(g_depths))); a.set_xticklabels(xt)
    a.set_yticks(range(len(g_leaves))); a.set_yticklabels(g_leaves)
    a.set_xlabel("max_depth"); a.set_ylabel("min_samples_leaf"); a.set_title(ttl)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            a.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7,
                   color="white" if Z[i, j] > Z.mean() else "black")
save("fig12_heatmaps.png")
pd.DataFrame(Z_te, index=g_leaves, columns=xt).round(2).to_csv("table_grid_test_rmse.csv")
pd.DataFrame(Z_gap, index=g_leaves, columns=xt).round(2).to_csv("table_grid_gap.csv")
RES["grid_min_rmse"] = float(Z_te.min())
bi, bj = np.unravel_index(np.argmin(Z_te), Z_te.shape)
RES["grid_best"] = {"leaf": g_leaves[bi], "depth": xt[bj]}
RES["grid_worst_gap"] = float(Z_gap.max())

# --- 6.4 Коректний підбір: GridSearchCV на навчальній вибірці ----------
grid_tree = GridSearchCV(
    DecisionTreeRegressor(random_state=14),
    {"max_depth": list(range(3, 21)),
     "min_samples_leaf": [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]},
    cv=cv5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_tree.fit(X_train, yr_train)
bt = grid_tree.best_estimator_
mb = tree_metrics(bt)
mf = tree_metrics(fit_tree(None, 1))
RES["tree_cv_best_params"] = grid_tree.best_params_
RES["tree_cv_best_rmse"] = float(-grid_tree.best_score_)
RES["tree_tuned"] = mb; RES["tree_full"] = mf
print("Найкращі параметри дерева:", grid_tree.best_params_, mb)

# візуалізація структури неглибокого дерева
plt.figure(figsize=(15, 6))
plot_tree(DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=14)
          .fit(X_train, yr_train), feature_names=features, filled=True, rounded=True, fontsize=8)
plt.title("Структура дерева регресії глибиною три")
save("fig13_tree_structure.png")

# --- 6.5 Подолання перенавчання ансамблем (бегінг/ліс) ------------------
mf_forest = rmse(yr_test, fitted_reg["Випадковий ліс"].predict(X_test))
RES["forest_rmse"] = mf_forest

# ======================================================================
# 7. ВАЖЛИВІСТЬ ОЗНАК: вбудована та пермутаційна
# ======================================================================
def importance_table(model_a, model_b, Xt, yt, scoring):
    perm = permutation_importance(model_b, Xt, yt, scoring=scoring, n_repeats=10,
                                  random_state=14, n_jobs=-1)
    return pd.DataFrame({
        "Випадковий ліс": model_a.feature_importances_,
        "Градієнтний бустинг": model_b.feature_importances_,
        "Пермутаційна важливість": perm.importances_mean}, index=features)

imp_r = importance_table(fitted_reg["Випадковий ліс"], fitted_reg["Градієнтний бустинг"],
                         X_test, yr_test, "r2").sort_values("Градієнтний бустинг")
imp_c = importance_table(fitted_cls["Випадковий ліс"], fitted_cls["Градієнтний бустинг"],
                         X_test, yc_test, "roc_auc").sort_values("Градієнтний бустинг")
for imp, nm, ttl in [(imp_r, "fig14_importance_reg.png", "Важливість ознак у задачі регресії"),
                     (imp_c, "fig15_importance_cls.png", "Важливість ознак у задачі класифікації")]:
    imp.plot(kind="barh", figsize=(10, 6.5)); plt.title(ttl); plt.xlabel("Важливість")
    plt.grid(True, axis="x", alpha=.4); save(nm)
imp_r.round(4).to_csv("table_importance_reg.csv"); imp_c.round(4).to_csv("table_importance_cls.csv")
print(imp_r.round(4)); print(imp_c.round(4))

# ======================================================================
# 8. ВПЛИВ КІЛЬКОСТІ БАЗОВИХ МОДЕЛЕЙ
# ======================================================================
stage_rmse = [rmse(yr_test, p) for p in fitted_reg["Градієнтний бустинг"].staged_predict(X_test)]
forest_sizes = [1, 5, 10, 25, 50, 100, 200, 400]
forest_rmse = []
for k in forest_sizes:
    rf = RandomForestRegressor(n_estimators=k, min_samples_leaf=5, n_jobs=-1, random_state=14)
    forest_rmse.append(rmse(yr_test, rf.fit(X_train, yr_train).predict(X_test)))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(range(1, len(stage_rmse) + 1), stage_rmse, color="#E45756")
ax[0].set_title("Градієнтний бустинг: RMSE від числа дерев")
ax[0].set_xlabel("Кількість дерев"); ax[0].set_ylabel("RMSE (тест)"); ax[0].grid(True)
ax[1].semilogx(forest_sizes, forest_rmse, "o-", color="#4C78A8")
ax[1].set_title("Випадковий ліс: RMSE від числа дерев")
ax[1].set_xlabel("Кількість дерев"); ax[1].set_ylabel("RMSE (тест)"); ax[1].grid(True)
save("fig16_n_estimators.png")

# ======================================================================
# 9. КРОС-ВАЛІДАЦІЯ ТА ПІДБІР ГІПЕРПАРАМЕТРІВ АНСАМБЛЮ
# ======================================================================
cv_s = StratifiedKFold(5, shuffle=True, random_state=14)
cvc = {"Логістична регресія": log_reg, "Дерево обмежене": tree_c_5,
       "Випадковий ліс": forest_c, "Градієнтний бустинг": gb_c,
       "Гістограмний бустинг": hgb_c}
cv_cls = {k: cross_val_score(m, X, y_cls, cv=cv_s, scoring="roc_auc", n_jobs=-1)
          for k, m in cvc.items()}
cvr = {"Лінійна регресія": lin, "Дерево обмежене": tree_r, "Випадковий ліс": forest_r,
       "Градієнтний бустинг": gb_r, "Гістограмний бустинг": hgb_r}
cv_reg = {k: cross_val_score(m, X, y_reg, cv=cv5, scoring="neg_root_mean_squared_error",
                             n_jobs=-1) * -1 for k, m in cvr.items()}
t_cv = pd.DataFrame(
    [{"Задача": "Класифікація (ROC-AUC)", "Модель": k, "Середнє": v.mean(), "Ст. відхилення": v.std()}
     for k, v in cv_cls.items()] +
    [{"Задача": "Регресія (RMSE)", "Модель": k, "Середнє": v.mean(), "Ст. відхилення": v.std()}
     for k, v in cv_reg.items()])
t_cv.round(4).to_csv("table_cv.csv", index=False); print(t_cv.round(4))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].boxplot(list(cv_cls.values()), tick_labels=list(cv_cls.keys()))
ax[0].set_title("ROC-AUC (5-кратна стратифікована CV)"); ax[0].tick_params(axis="x", rotation=20)
ax[1].boxplot(list(cv_reg.values()), tick_labels=list(cv_reg.keys()))
ax[1].set_title("RMSE (5-кратна CV)"); ax[1].tick_params(axis="x", rotation=20)
for a in ax: a.grid(True, axis="y", alpha=.4)
save("fig17_cv.png")

grid_rf = GridSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=14),
    {"n_estimators": [100, 300], "max_depth": [8, 12, None], "min_samples_leaf": [1, 5, 20]},
    cv=3, scoring="neg_root_mean_squared_error")
grid_rf.fit(X_train, yr_train)
brf = grid_rf.best_estimator_
RES["rf_best_params"] = grid_rf.best_params_
RES["rf_cv_rmse"] = float(-grid_rf.best_score_)
RES["rf_test_rmse"] = rmse(yr_test, brf.predict(X_test))
RES["rf_test_r2"] = float(r2_score(yr_test, brf.predict(X_test)))
print("RF:", RES["rf_best_params"], RES["rf_cv_rmse"], RES["rf_test_rmse"], RES["rf_test_r2"])

grid_rfc = GridSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=14),
    {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 5, 20]},
    cv=3, scoring="roc_auc")
grid_rfc.fit(X_train, yc_train)
brfc = grid_rfc.best_estimator_
RES["rfc_best_params"] = grid_rfc.best_params_
RES["rfc_cv_auc"] = float(grid_rfc.best_score_)
RES["rfc_test_auc"] = float(roc_auc_score(yc_test, brfc.predict_proba(X_test)[:, 1]))
RES["rfc_test_f1"] = float(f1_score(yc_test, brfc.predict(X_test)))
print("RFC:", RES["rfc_best_params"], RES["rfc_cv_auc"], RES["rfc_test_auc"], RES["rfc_test_f1"])

# ======================================================================
# 10. ПРОГНОЗ ДЛЯ НОВИХ ОБ'ЄКТІВ (кінцевий конвеєр)
# ======================================================================
final_reg = Pipeline([("scaler", StandardScaler()),
                      ("model", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                          max_depth=3, subsample=0.8, random_state=14))])
final_cls = Pipeline([("scaler", StandardScaler()),
                      ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                               max_depth=4, random_state=14))])
final_reg.fit(X_train, yr_train); final_cls.fit(X_train, yc_train)
new = pd.DataFrame({
    "area": [45, 140], "floors": [5, 2], "building_age": [60, 5], "insulation": [2.5, 9],
    "window_ratio": [0.40, 0.20], "heating_temp": [-5, 5], "residents": [5, 2],
    "appliances": [13, 6], "noise_1": [0, 0], "noise_2": [0, 0], "noise_3": [0, 0]})
RES["new_cls"] = final_cls.predict_proba(new)[:, 1].round(3).tolist()
RES["new_reg"] = final_reg.predict(new).round(1).tolist()
print(RES["new_cls"], RES["new_reg"])

# ======================================================================
# Збереження підсумкових результатів
# ======================================================================
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return str(o)
RES["forest_sizes"] = forest_sizes; RES["forest_rmse"] = forest_rmse
RES["stage_rmse_min"] = float(min(stage_rmse)); RES["stage_rmse_argmin"] = int(np.argmin(stage_rmse)) + 1
RES["stage_rmse_last"] = float(stage_rmse[-1])
json.dump(RES, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=conv)
print("ГОТОВО")
