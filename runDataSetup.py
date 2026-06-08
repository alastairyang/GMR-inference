import numpy as np
import os
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy
sys.path.append('../')
from src.ice import compute_pmp
from src.utilities import cluster_points, build_masks_from_points

# ---------------- PARAMETERS ----------------
train_ratio      = 0.9
validation_ratio = 0
test_ratio       = 0.1
n_folds          = 10

data_path  = os.getenv('DATA_PATH')
param_path = os.getenv('PARAM_PATH')
folder = data_path
# ------------------------------------------------- LOAD SIMULATION DATA -------------------------------------------------
# get the DATA_PATH from the environment variable
Eb_filename           = 'trainingAll4D_Eb_sim_standardized.mat'
Ns_filename           = 'trainingAll4D_Ns_sim_masked_standardized.mat'
GHF_filename          = 'trainingAll4D_GHF_sim_standardized.mat'
coord_filename        = "trainingAll_image_coord.mat"

Eb_data           = scipy.io.loadmat(folder + Eb_filename)
Ns_data           = scipy.io.loadmat(folder + Ns_filename)
GHF_data          = scipy.io.loadmat(folder + GHF_filename)
coord_data        = scipy.io.loadmat(folder + coord_filename)

# load
Eb_standardized  = Eb_data['Eb_standardized']
Ns_standardized  = Ns_data['Ns_standardized_masked']
GHF_standardized = GHF_data['GHF_standardized']
# -------------------
assert train_ratio + validation_ratio + test_ratio == 1.0, "The sum of the ratios must be 1."

n_samples_total = Eb_standardized.shape[-1]
n_train = int(n_samples_total * train_ratio)
n_validation = int(n_samples_total * validation_ratio)
n_test = n_samples_total - n_train - n_validation

indices = np.arange(n_samples_total)
random_state = np.random.RandomState(seed=42)
random_state.shuffle(indices)

# print the first 20 randomly shuffled indices for verification
print("First 20 randomly shuffled indices:", indices[:20])

Eb_train      = Eb_standardized[..., indices[:n_train]]
Eb_validation = Eb_standardized[..., indices[n_train:n_train + n_validation]]
Eb_test       = Eb_standardized[..., indices[n_train + n_validation:]]

Ns_train      = Ns_standardized[..., indices[:n_train]]
Ns_validation = Ns_standardized[..., indices[n_train:n_train + n_validation]]
Ns_test       = Ns_standardized[..., indices[n_train + n_validation:]]

GHF_train      = GHF_standardized[..., indices[:n_train]]
GHF_validation = GHF_standardized[..., indices[n_train:n_train + n_validation]]
GHF_test       = GHF_standardized[..., indices[n_train + n_validation:]]

print("Shape of Eb_train:", Eb_train.shape)
print("Shape of Eb_validation:", Eb_validation.shape)
print("Shape of Eb_test:", Eb_test.shape)
print("Shape of Ns_train:", Ns_train.shape)
print("Shape of Ns_validation:", Ns_validation.shape)
print("Shape of Ns_test:", Ns_test.shape)
print("Shape of GHF_train:", GHF_train.shape)
print("Shape of GHF_validation:", GHF_validation.shape)
print("Shape of GHF_test:", GHF_test.shape)

save_folder = "data/train-validate-test/"
os.makedirs(save_folder, exist_ok=True)
np.savez(save_folder + "train_data.npz", Eb_train=Eb_train, Ns_train=Ns_train, GHF_train=GHF_train)
np.savez(save_folder + "validation_data.npz", Eb_validation=Eb_validation, Ns_validation=Ns_validation, GHF_validation=GHF_validation)
np.savez(save_folder + "test_data.npz", Eb_test=Eb_test, Ns_test=Ns_test, GHF_test=GHF_test)
print(f"Data saved to {save_folder}")

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# --- Build the training/validation and test set for the basal thermal evidence ----
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


H_filename           = folder + 'H_gridded.mat'
Eb_mean_filename     = folder + 'trainingAll_Eb_sim_mean.mat'
Eb_std_filename      = folder + 'trainingAll_Eb_sim_std.mat'
frozen_base_filename = folder + "../../basal-frozen-mask/frozen_mask.mat"
thawed_base_filename = folder + "../../basal-water-mask/water_mask.mat"

H_data       = scipy.io.loadmat(H_filename)
Eb_mean_data = scipy.io.loadmat(Eb_mean_filename)
Eb_std_data  = scipy.io.loadmat(Eb_std_filename)

# load the thawed base and frozen base data
frozen_mask = scipy.io.loadmat(frozen_base_filename)['frozen_mask']
thawed_mask = scipy.io.loadmat(thawed_base_filename)['water_mask']
frozen_mask_data = frozen_mask['data'][0][0]
thawed_mask_data = thawed_mask['data'][0][0]

# replace nan by 0
Eb_mean_data = np.nan_to_num(Eb_mean_data['Eb_mean'], nan=0.0)
Eb_std_data  = np.nan_to_num(Eb_std_data['Eb_std'], nan=0.0)

# compute pmp
H = H_data['H_struct']['H'][0][0].flatten()
X = H_data['H_struct']['X'][0][0].flatten()
Y = H_data['H_struct']['Y'][0][0].flatten()
pmp = compute_pmp(H)
# get coordinate
x_coord = coord_data['training_coord'][0][0][0]
y_coord = coord_data['training_coord'][0][0][1]
X_grid, Y_grid = np.meshgrid(x_coord, y_coord)

Eb_mean_data     = Eb_mean_data.flatten()
Eb_std_data      = Eb_std_data.flatten()
frozen_mask_data = frozen_mask_data.flatten()
thawed_mask_data = thawed_mask_data.flatten()

# set fractional area to 1 for all locations (assumption for now, can be relaxed later)
thawed_fractional_area = np.zeros_like(thawed_mask_data)
frozen_fractional_area = np.zeros_like(frozen_mask_data)
thawed_fractional_area[thawed_mask_data == 1] = 1.0
frozen_fractional_area[frozen_mask_data == 1] = 1.0

thawed_x = X_grid.flatten()[thawed_mask_data == 1]
frozen_x = X_grid.flatten()[frozen_mask_data == 1]
thawed_y = Y_grid.flatten()[thawed_mask_data == 1]
frozen_y = Y_grid.flatten()[frozen_mask_data == 1]
idx_thawed = np.where(thawed_mask_data == 1)[0]
idx_frozen = np.where(frozen_mask_data == 1)[0]
evidence_x = np.concatenate([thawed_x, frozen_x])
evidence_y = np.concatenate([thawed_y, frozen_y])
print("shape of evidence_x:", evidence_x.shape)
print("shape of evidence_y:", evidence_y.shape)

# CLUSTER ALL EVIDENCE INTO 50 GROUPS
cluster_labels, cluster_centroids, cluster_groups = cluster_points(
    evidence_x, evidence_y, n=50, show_plot=False
)

# SPLIT: 5 HOLDOUT CLUSTERS vs 45 TRAINING CLUSTERS 
n_ppc_clusters = 8
np.random.seed(42)
ppc_cluster_indices = set(np.random.choice(50, size=n_ppc_clusters, replace=False))
train_cluster_indices = [i for i in range(50) if i not in ppc_cluster_indices]

print("Holdout cluster indices (PPC):", sorted(ppc_cluster_indices))
print("Training cluster indices     :", train_cluster_indices)

# build thawed/frozen masks from a list of (x,y) points 
flat_x = X_grid.flatten()
flat_y = Y_grid.flatten()
grid_shape = (X_grid.shape[0], X_grid.shape[1])

# BUILD HOLDOUT (PPC) MASKS 
ppc_points = [pt for i in ppc_cluster_indices for pt in cluster_groups[i]]

ppc_thawed_mask, ppc_frozen_mask = build_masks_from_points(
    ppc_points, idx_thawed, idx_frozen, flat_x, flat_y, grid_shape
)

ppc_thawed_area = np.zeros_like(ppc_thawed_mask, dtype=float)
ppc_frozen_area = np.zeros_like(ppc_frozen_mask, dtype=float)
ppc_thawed_area[ppc_thawed_mask] = 1.0
ppc_frozen_area[ppc_frozen_mask] = 1.0

ppc_evidence_mask = {
    "ppc_thawed_mask" : ppc_thawed_mask,
    "ppc_frozen_mask" : ppc_frozen_mask
}
ppc_evidence_area = {
    "ppc_thawed_area" : ppc_thawed_area,
    "ppc_frozen_area" : ppc_frozen_area
}

print("PPC thawed points :", ppc_thawed_mask.sum())
print("PPC frozen points :", ppc_frozen_mask.sum())

# BUILD TRAINING (45-CLUSTER) MASKS 
train_points = [pt for i in train_cluster_indices for pt in cluster_groups[i]]

train_thawed_mask, train_frozen_mask = build_masks_from_points(
    train_points, idx_thawed, idx_frozen, flat_x, flat_y, grid_shape
)

print("Training thawed points:", train_thawed_mask.sum())
print("Training frozen points:", train_frozen_mask.sum())

# RE-CLUSTER TRAINING POINTS INTO 10 CV FOLDS
train_x = flat_x[train_thawed_mask.flatten() | train_frozen_mask.flatten()]
train_y = flat_y[train_thawed_mask.flatten() | train_frozen_mask.flatten()]

cv_labels, cv_centroids, cv_groups = cluster_points(
    train_x, train_y, n=n_folds, show_plot=False
)

# BUILD 10 SETS OF (thawed_mask, frozen_mask) FOR CV FOLDS
cv_masks = []  # list of 10 dicts, each with 'train' and 'val' masks
cv_areas = []  # fractional area

for fold_idx in range(n_folds):
    val_points   = cv_groups[fold_idx]
    train_fold_points = [pt for j, pts in cv_groups.items()
                         if j != fold_idx for pt in pts]

    val_thawed,   val_frozen   = build_masks_from_points(
        val_points, idx_thawed, idx_frozen, flat_x, flat_y, grid_shape
    )
    train_thawed, train_frozen = build_masks_from_points(
        train_fold_points, idx_thawed, idx_frozen, flat_x, flat_y, grid_shape
    )

    cv_masks.append({
        "fold"         : fold_idx,
        "val_thawed"   : val_thawed,
        "val_frozen"   : val_frozen,
        "train_thawed" : train_thawed,
        "train_frozen" : train_frozen,
    })

    val_thawed_area = np.zeros_like(val_thawed, dtype=float)
    val_frozen_area = np.zeros_like(val_frozen, dtype=float)
    val_thawed_area[val_thawed] = 1.0
    val_frozen_area[val_frozen] = 1.0
    train_thawed_area = np.zeros_like(train_thawed, dtype=float)
    train_frozen_area = np.zeros_like(train_frozen, dtype=float)
    train_thawed_area[train_thawed] = 1.0
    train_frozen_area[train_frozen] = 1.0
    cv_areas.append({
        "fold"         : fold_idx,
        "val_thawed_area"   : val_thawed_area,
        "val_frozen_area"   : val_frozen_area,
        "train_thawed_area" : train_thawed_area,
        "train_frozen_area" : train_frozen_area,
    })

    print(f"Fold {fold_idx:2d} | "
          f"train thawed={train_thawed.sum():4d}  frozen={train_frozen.sum():4d} | "
          f"val   thawed={val_thawed.sum():4d}  frozen={val_frozen.sum():4d}")

# save PPC masks and areas
ppc_save_folder = save_folder
os.makedirs(ppc_save_folder, exist_ok=True)
np.savez(ppc_save_folder + "ppc_evidence_masks.npz", **ppc_evidence_mask)
np.savez(ppc_save_folder + "ppc_evidence_areas.npz", **ppc_evidence_area)
print(f"PPC evidence masks and areas saved to {ppc_save_folder}")

cv_save_folder = save_folder
os.makedirs(cv_save_folder, exist_ok=True)
for fold_idx in range(n_folds):
    fold_masks = cv_masks[fold_idx]
    fold_areas = cv_areas[fold_idx]
    np.savez(cv_save_folder + f"cv_fold_{fold_idx}_masks.npz", **fold_masks)
    np.savez(cv_save_folder + f"cv_fold_{fold_idx}_areas.npz", **fold_areas)
print(f"CV fold masks and areas saved to {cv_save_folder}")

# visualize the cv masks
def plot_folds(cv_masks, flat_x, flat_y, figsize=(24, 12), save_path=None):
    """
    Plot all 10 CV folds as subplots.
    Each subplot shows:
        - Grey  : points not in this fold's train or val set (PPC holdout)
        - Blue  : training points for this fold
        - Red   : validation points for this fold
        - Green : thawed points
        - Orange: frozen points
    """
    n_folds = len(cv_masks)
    ncols = 5
    nrows = (n_folds + ncols - 1) // ncols 

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for fold_idx, fold in enumerate(cv_masks):
        ax = axes[fold_idx]

        train_thawed = fold["train_thawed"].flatten()
        train_frozen = fold["train_frozen"].flatten()
        val_thawed   = fold["val_thawed"].flatten()
        val_frozen   = fold["val_frozen"].flatten()

        # ── plot training points ──────────────────────────────────────────
        ax.scatter(
            flat_x[train_thawed], flat_y[train_thawed],
            c="steelblue", s=8, alpha=0.6, label="Train – Thawed"
        )
        ax.scatter(
            flat_x[train_frozen], flat_y[train_frozen],
            c="cornflowerblue", s=8, alpha=0.6, marker="s", label="Train – Frozen"
        )

        # ── plot validation points ────────────────────────────────────────
        ax.scatter(
            flat_x[val_thawed], flat_y[val_thawed],
            c="tomato", s=18, alpha=0.9, zorder=3, label="Validate – Thawed"
        )
        ax.scatter(
            flat_x[val_frozen], flat_y[val_frozen],
            c="firebrick", s=18, alpha=0.9, zorder=3, marker="s", label="Validate – Frozen"
        )

        n_train = train_thawed.sum() + train_frozen.sum()
        n_val   = val_thawed.sum()   + val_frozen.sum()

        ax.set_title(f"Fold {fold_idx}  |  train={n_train}  validate={n_val}",
                     fontsize=15)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_aspect("equal", adjustable="box")

    for j in range(n_folds, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="steelblue",      label="Train – Thawed"),
        mpatches.Patch(color="cornflowerblue", label="Train – Frozen"),
        mpatches.Patch(color="tomato",         label="Validate – Thawed"),
        mpatches.Patch(color="firebrick",      label="Validate – Frozen"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=18,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle("K-Fold CV Splits (10 Folds) — Spatial Clusters",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CV fold plot saved to {save_path}")


plot_folds(cv_masks,  
           flat_x,  flat_y, 
           save_path="figs/parameter-tuning/cv_masks.png")

# Visualize the holdout dataset, with the K-fold datasets grayed out in the background
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ppc_thawed = ppc_evidence_mask["ppc_thawed_mask"].flatten()
ppc_frozen  = ppc_evidence_mask["ppc_frozen_mask"].flatten()

# ── non-holdout points (grey background) ─────────────────────────────────────
# thawed that are NOT in holdout → dark grey
bg_thawed = idx_thawed[~ppc_thawed[idx_thawed]]
# frozen that are NOT in holdout → light grey
bg_frozen  = idx_frozen[~ppc_frozen[idx_frozen]]

ax.scatter(
    flat_x[bg_thawed], flat_y[bg_thawed],
    c="#555555", s=8, alpha=0.4, zorder=1, label="Training – Thawed"
)
ax.scatter(
    flat_x[bg_frozen], flat_y[bg_frozen],
    c="#aaaaaa", s=8, alpha=0.4, zorder=1, marker="s", label="Training – Frozen"
)

# ── holdout points (coloured foreground) ─────────────────────────────────────
ax.scatter(
    flat_x[ppc_thawed], flat_y[ppc_thawed],
    c="tomato", s=10, alpha=0.4, zorder=3, label="Holdout – Thawed"
)
ax.scatter(
    flat_x[ppc_frozen], flat_y[ppc_frozen],
    c="firebrick", s=10, alpha=0.4, zorder=3, marker="s", label="Holdout – Frozen"
)

# ── formatting ────────────────────────────────────────────────────────────────
n_ppc = ppc_thawed.sum() + ppc_frozen.sum()

ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.tick_params(labelsize=10)
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=9, markerscale=1.5, framealpha=0.8)

fig.suptitle(f"Holdout Set for Posterior Predictive Check (PPC)  [n={n_ppc}]",
             fontsize=14, y=1.01)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("figs/parameter-tuning/ppc_holdout.png", dpi=300, bbox_inches="tight")
print(f"PPC holdout plot saved to figs/parameter-tuning/ppc_holdout.png")