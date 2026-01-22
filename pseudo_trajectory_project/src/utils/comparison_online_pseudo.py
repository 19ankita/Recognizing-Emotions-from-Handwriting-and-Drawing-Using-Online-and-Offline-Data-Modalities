import os
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
TASKS = ["cdt", "cursive_writing", "house", "pentagon", "words"]

ONLINE_ROOT = "data/EMOTHAW/online_features"
PSEUDO_ROOT = "data/EMOTHAW/pseudo_features"

OUT_CSV = "data/EMOTHAW/online_pseudo_summary.csv"

COMPARABLE_FEATURES = [
    "path_length",
    "straightness",
    "dominant_angle",
    "direction_concentration",
    "width",
    "height",
    "aspect_ratio",
    "median_speed",
    "p95_speed",
]


# =========================
# HELPERS
# =========================
def safe_mean(x):
    return np.nanmean(x)


def recoverability_score(mu_online, mu_pseudo):
    """
    Symmetric ratio in [0, 1]
    """
    if mu_online == 0 or mu_pseudo == 0:
        return 0.0
    return min(mu_online, mu_pseudo) / max(mu_online, mu_pseudo)


# =========================
# MAIN
# =========================
rows = []

for task in TASKS:
    print(f"[INFO] Comparing ONLINE vs PSEUDO for {task}")

    online_path = os.path.join(ONLINE_ROOT, f"{task}_features.csv")
    pseudo_path = os.path.join(PSEUDO_ROOT, f"{task}_pseudo_features.csv")

    if not os.path.exists(online_path):
        print(f"  [SKIP] Missing online file: {online_path}")
        continue

    if not os.path.exists(pseudo_path):
        print(f"  [SKIP] Missing pseudo file: {pseudo_path}")
        continue

    online = pd.read_csv(online_path)
    pseudo = pd.read_csv(pseudo_path)

    for feat in COMPARABLE_FEATURES:
        if feat not in online.columns or feat not in pseudo.columns:
            continue

        mu_online = safe_mean(online[feat])
        mu_pseudo = safe_mean(pseudo[feat])

        std_online = np.nanstd(online[feat])
        std_pseudo = np.nanstd(pseudo[feat])

        rec = recoverability_score(mu_online, mu_pseudo)

        rows.append({
            "task": task,
            "feature": feat,
            "online_mean": mu_online,
            "pseudo_mean": mu_pseudo,
            "online_std": std_online,
            "pseudo_std": std_pseudo,
            "recoverability": rec,
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"\n Recoverability table saved to: {OUT_CSV}")
