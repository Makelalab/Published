"""Localization heat map in normalized cell coordinates (folded half cell).

Input: a cleaned track CSV (columns fov_mask_id, cell_id_int, x_t, y_t, Dapp_log)
and the Omnipose masks of the same fields of view.

For every cell: principal axes from the mask pixels -> localizations projected
onto them -> divided by the cell half-length (so the pole is at -1, the centre at 0)
-> long axis folded (-|pc0|) because the two poles cannot be told apart.
All cells are pooled into one 2D histogram (fraction of localizations per bin).
The black outline is the average cell shape, built the same way from the mask pixels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "svg.fonttype": "none"})

PX = 0.106          # µm per pixel
MAX_AREA = 2.5      # µm²; larger cells are excluded (as in the original analysis)
BIN = 0.08          # bin width in normalized units (same in x and y)
Y_MAX = 0.6         # width axis shown from -Y_MAX to +Y_MAX
X_MIN = -1.12       # length axis shown from X_MIN to 0
OUTLINE_BIN = 0.01      # outline is computed on this finer grid ...
OUTLINE_SMOOTH = 0.05   # ... and smoothed this much (normalized units). Larger = smoother.
CMAP = "magma_r"     # light = few, dark = many localizations


def _load_mask(path):
    d = np.load(path, allow_pickle=True)
    if d.dtype == object:                 # Omnipose *_seg.npy (dict)
        d = d.item()["masks"]
    return np.asarray(d)


def _mask_path(masks, fov_mask_id):
    if isinstance(masks, dict):
        return masks[fov_mask_id]
    return Path(masks) / fov_mask_id      # folder with files named like fov_mask_id


def to_cell_coords(csv, masks, max_area=MAX_AREA, dapp_range=None):
    """Return (localizations, mask pixels) in folded, normalized cell coordinates."""
    df = pd.read_csv(csv)
    if dapp_range is not None:
        lo, hi = dapp_range
        df = df[(df["Dapp_log"] > lo) & (df["Dapp_log"] < hi)]
    locs, pix, n_cells = [], [], 0
    for fov, d_fov in df.groupby("fov_mask_id"):
        M = _load_mask(_mask_path(masks, fov))
        for cell, d in d_fov.groupby("cell_id_int"):
            r, c = np.nonzero(M == cell)
            if len(r) < 10 or len(r) * PX**2 >= max_area:
                continue
            P = np.c_[r, c] + 0.5                       # pixel centres
            mu = P.mean(0)
            _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
            pm = (P - mu) @ vt.T
            half = np.abs(pm[:, 0]).max()               # cell half-length (pixels)
            pl = (d[["x_t", "y_t"]].to_numpy() - mu) @ vt.T
            locs.append(pl / half)
            pix.append(pm / half)
            n_cells += 1
    locs, pix = np.vstack(locs), np.vstack(pix)
    locs[:, 0], pix[:, 0] = -np.abs(locs[:, 0]), -np.abs(pix[:, 0])
    return locs, pix, n_cells


def plot_heatmap(csv, masks, title=None, vmax=None, dapp_range=None,
                 save=None, ax=None, colorbar=True):
    locs, pix, n_cells = to_cell_coords(csv, masks, dapp_range=dapp_range)
    ny = int(round(Y_MAX / BIN))
    yedges = np.linspace(-ny * BIN, ny * BIN, 2 * ny + 1)
    xedges = np.arange(-np.ceil(-X_MIN / BIN) * BIN, BIN / 2, BIN)

    H = np.histogram2d(locs[:, 0], locs[:, 1], [xedges, yedges])[0]
    H = H / H.sum()                                     # fraction of localizations
    S = gaussian_filter(np.histogram2d(pix[:, 0], pix[:, 1], [xedges, yedges])[0],
                        0.5, mode="reflect")
    S[-1, :] = S[-2, :]
    thr = threshold_otsu(S[S > 0]) * 0.5                # average cell outline
    inside = S >= thr

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.0, 2.6))
    else:
        fig = ax.figure
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("white")
    img = H.T   # image is clipped to the cell outline below
    im = ax.imshow(img, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   origin="lower", cmap=cmap, vmin=0,
                   vmax=vmax if vmax else img.max(), interpolation="nearest")
    xc, yc = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    # smooth outline: same rule (Otsu x 0.5) on a fine, smoothed grid of mask pixels
    fx = np.arange(xedges[0], OUTLINE_BIN / 2, OUTLINE_BIN)
    fy = np.arange(yedges[0], yedges[-1] + OUTLINE_BIN / 2, OUTLINE_BIN)
    F = np.histogram2d(pix[:, 0], pix[:, 1], [fx, fy])[0]
    F = gaussian_filter(F, OUTLINE_SMOOTH / OUTLINE_BIN, mode="reflect")
    fthr = threshold_otsu(F[F > 0]) * 0.5
    cs = ax.contour((fx[:-1] + fx[1:]) / 2, (fy[:-1] + fy[1:]) / 2, F.T,
                    levels=[fthr], colors="black", linewidths=1.0)
    # close the outline along the cell centre (x = 0) and clip the image to it
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    v = max(cs.allsegs[0], key=len)
    v = np.vstack([[0, v[0, 1]], v, [0, v[-1, 1]]])
    clip = PathPatch(MPath(v), transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    ax.set_aspect("equal")
    ax.set_xlim(xedges[0], 0)
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_xlabel("Normalized cell length")
    ax.set_ylabel("Normalized cell width")
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold")
    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Fraction of localizations")
        cb.outline.set_visible(False)
    if save:
        fig.savefig(save, bbox_inches="tight")
    print(f"{Path(csv).name}: {len(locs)} localizations, {n_cells} cells, "
          f"max bin = {H[inside].max():.4f}")
    return fig, ax, im
