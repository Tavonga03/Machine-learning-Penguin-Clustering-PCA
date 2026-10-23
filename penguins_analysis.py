

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("penguins_uts.csv")

print("Shape:", df.shape) #printing the shape and the size of the dataset
print("\nFirst rows:")
print(df.head())
print("\nInfo:")
df.info()

# ---------------------------------------------------------------
# 2. Data quality checks
# ---------------------------------------------------------------
print("\nMissing values per column:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nCategory counts:")
for col in ["species", "island", "sex"]:
    print(f"\n{col}:\n{df[col].value_counts(dropna=False)}")

# ---------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------
num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# Drop rows missing any measurement; keep 'sex' NaNs out of sex-specific analysis only
clean = df.dropna(subset=num_cols).copy()
clean["sex"] = clean["sex"].str.lower().str.strip()
print(f"\nRows after dropping missing measurements: {len(clean)} (from {len(df)})")

# ---------------------------------------------------------------
# 4. Descriptive statistics
# ---------------------------------------------------------------
print("\nOverall summary:")
print(clean[num_cols].describe().round(2))

print("\nMean measurements by species:")
print(clean.groupby("species")[num_cols].mean().round(2))

print("\nMean measurements by species and sex:")
print(clean.dropna(subset=["sex"]).groupby(["species", "sex"])[num_cols].mean().round(2))

print("\nSpecies by island:")
print(pd.crosstab(clean["island"], clean["species"]))

# ---------------------------------------------------------------
# 5. Visualisations
# ---------------------------------------------------------------
# 5a. Distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(data=clean, x=col, hue="species", kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("distributions.png", dpi=150)

# 5b. Boxplots by species
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.boxplot(data=clean, x="species", y=col, ax=ax)
    ax.set_title(f"{col} by species")
plt.tight_layout()
plt.savefig("boxplots.png", dpi=150)

# 5c. Pairplot
pair = sns.pairplot(clean, vars=num_cols, hue="species", corner=True)
pair.savefig("pairplot.png", dpi=120)

# 5d. Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(clean[num_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("correlation.png", dpi=150)

# 5e. Body mass by species and sex
plt.figure(figsize=(7, 5))
sns.boxplot(data=clean.dropna(subset=["sex"]), x="species", y="body_mass_g", hue="sex")
plt.title("Body mass by species and sex")
plt.tight_layout()
plt.savefig("mass_by_species_sex.png", dpi=150)

# 5f. Flipper length vs body mass
plt.figure(figsize=(7, 5))
sns.scatterplot(data=clean, x="flipper_length_mm", y="body_mass_g",
                hue="species", style="island")
plt.title("Flipper length vs body mass")
plt.tight_layout()
plt.savefig("flipper_vs_mass.png", dpi=150)

# ---------------------------------------------------------------
# 6. Classification: predict species from measurements
# ---------------------------------------------------------------
X = clean[num_cols]
y = clean["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}

for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"5-fold CV accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")
    print(classification_report(y_test, preds))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds, labels=model.classes_),
        display_labels=model.classes_,
    ).plot(cmap="Blues")
    plt.title(f"Confusion matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)

# Feature importance from the random forest
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=num_cols).sort_values()
plt.figure(figsize=(6, 4))
importance.plot(kind="barh")
plt.title("Random Forest feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)

print("\nDone. Plots saved as PNG files in the current folder.")
plt.show()
