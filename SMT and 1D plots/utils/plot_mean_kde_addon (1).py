"""Independent 1D plotting functions for the existing plot_test.py workflow.

Import this module next to your notebook, or append its imports and function
definitions to the end of plot_test.py. Existing functions need no changes.
There is no import-time plotting, file access, or modification of input data.

Inputs may be original trajectory tables or tables already containing pc0/pc1.
Raw tables use the original project's utils.get_features.to_PC_raw and get_area;
run from that project so its helpers and any resources they need are available.
No replacement PCA implementation is used.
The default geometric references use a capped-cylinder long-axis shape for
PC0 and the projection of a uniformly filled disk (circle chord) for PC1.
By default the PC0 reference has area 1 over [-1, 0], matching the folded
histogram. Its symmetric extension to [0, 1] gives a full displayed area of 2.
Reference smoothing is optional and off by default.
Dependencies: numpy, pandas, scipy, seaborn, matplotlib.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.integrate import trapezoid


def _pmk_feature_filter(df, dapp_range):
    """Apply exclusive Dapp_log bounds, without modifying the input."""
    if dapp_range is None:
        return df.copy()
    if len(dapp_range) != 2:
        raise ValueError("dapp_range must be (lower, upper), or None.")
    lower, upper = dapp_range
    if not lower < upper:
        raise ValueError("The lower Dapp_log limit must be below the upper limit.")
    if "Dapp_log" not in df.columns:
        raise ValueError("Dapp_log is required for the requested filter.")
    values = pd.to_numeric(df["Dapp_log"], errors="raise")
    return df.loc[(values > lower) & (values < upper)].copy()


def _pmk_project_helpers(pc_transform, area_func, need_area):
    """Load only the original helpers needed for raw-coordinate conversion."""
    if pc_transform is None or (need_area and area_func is None):
        try:
            from utils import get_features
        except ImportError as exc:
            raise ImportError(
                "Raw CSV input requires the original project's "
                "utils.get_features module and its dependencies. Run this "
                "notebook from the original project root, or supply "
                "pc_transform=to_PC_raw and area_func=get_area explicitly. "
                "The add-on does not replace the original coordinate transform."
            ) from exc
        if pc_transform is None:
            pc_transform = get_features.to_PC_raw
        if need_area and area_func is None:
            area_func = get_features.get_area
    if not callable(pc_transform) or (need_area and not callable(area_func)):
        raise TypeError("pc_transform and the required area_func must be callable.")
    return pc_transform, area_func


def prepare_pc_coordinates(
    df, *, dapp_range=None, pixel_um=0.106, max_area=2.5,
    pc_groupby="track", experiment_id=None, pc_transform=None, area_func=None,
):
    """Copy an input table and return coordinates suitable for the new plots.

    Already-PC input: keep pc0/pc1 unchanged and apply only dapp_range, if set.
    Raw input: follow spatial_distribution's Dapp_log preparation workflow:
    rename x_t/y_t to x_t_px/y_t_px; use the first Dapp_log value per track;
    calculate cell areas with the original get_area; keep area < max_area;
    apply lower < Dapp_log < upper; call the original to_PC_raw with
    pixel=pixel_um and absPC=False on the remaining groups.

    The original spatial_distribution groups the PC conversion by experiment,
    mask_uuid, cell_id_int AND particle. Accordingly, pc_groupby='track' is
    the default. To follow density_for_feat_range's Dapp_log workflow instead,
    choose pc_groupby='cell' and max_area=None. These are different original
    code paths; the add-on does not silently change the grouping.

    Raw tables need x_t/y_t (or x_t_px/y_t_px), mask_uuid, cell_id_int,
    particle, fov_mask_id, and comment, as used in the original code. Any extra
    columns are preserved. Dapp_log is required only if dapp_range is provided.
    An existing experiment column is preserved unless experiment_id is set;
    otherwise a missing experiment column is filled with 'experiment0'.

    pc_transform and area_func may be the original project's imported helper
    functions. By default they are imported lazily from utils.get_features.
    A transform must return one two-column row per input row, in input order
    or indexed by the input row index. Group keys remain available as columns
    and as group.name. Input row order is retained in the returned table.

    The returned PC coordinates are NOT folded. plot_pc_mean_kde applies
    -abs(pc0) later on a separate copy when fold_pc0=True.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if {"pc0", "pc1"}.issubset(df.columns):
        selected = _pmk_feature_filter(df, dapp_range)
        if selected.empty:
            raise ValueError("No localizations remain after Dapp_log filtering.")
        return selected
    if "pc0" in df.columns or "pc1" in df.columns:
        raise ValueError("Only one PC column is present; provide both, or neither for raw input.")
    if pc_groupby not in ("track", "cell"):
        raise ValueError("pc_groupby must be 'track' or 'cell'.")
    if not np.isfinite(pixel_um) or pixel_um <= 0:
        raise ValueError("pixel_um must be positive and finite.")
    if max_area is not None and (not np.isfinite(max_area) or max_area <= 0):
        raise ValueError("max_area must be positive and finite, or None.")

    # An internal RangeIndex makes assignment safe even for shuffled or repeated
    # caller indices; the original index is restored on the returned table.
    source_index = df.index.copy()
    raw = df.copy().reset_index(drop=True)
    for old, new in (("x_t", "x_t_px"), ("y_t", "y_t_px")):
        if old in raw.columns and new in raw.columns:
            raise ValueError(f"Both {old} and {new} are present; keep one coordinate column.")
        if old in raw.columns:
            raw.rename(columns={old: new}, inplace=True)
    required = ["x_t_px", "y_t_px", "mask_uuid", "cell_id_int", "particle",
                "fov_mask_id", "comment"]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError("Raw-coordinate input is missing columns used by the original "
                         "workflow: " + ", ".join(missing))
    if raw.empty:
        raise ValueError("The input table contains no localizations.")
    if experiment_id is not None:
        raw["experiment"] = str(experiment_id)
    elif "experiment" not in raw.columns:
        raw["experiment"] = "experiment0"
    raw["fov_uids"] = raw["experiment"].astype(str) + "_" + raw["comment"].astype(str)
    cell_keys = ["experiment", "mask_uuid", "cell_id_int"]
    track_keys = cell_keys + ["particle"]
    if raw[track_keys].isna().any().any():
        raise ValueError("Raw-coordinate grouping columns must not contain missing values.")
    if "Dapp_log" not in raw.columns and dapp_range is not None:
        raise ValueError("Dapp_log is required for the requested filter.")

    pc_transform, area_func = _pmk_project_helpers(
        pc_transform, area_func, need_area=max_area is not None
    )
    if max_area is not None:
        areas = pd.Series(index=raw.index, dtype=float)
        for key, group in raw.groupby(cell_keys, sort=False):
            group_rows = group.index.copy()
            group = group.copy()
            object.__setattr__(group, "name", key)
            result = np.asarray(area_func(group, pixel=pixel_um), dtype=float)
            if result.size != 1:
                raise ValueError("get_area must return one cell-area value per group.")
            areas.loc[group_rows] = float(result.ravel()[0])
        raw["area"] = areas
    if "Dapp_log" in raw.columns:
        raw["Dapp_log"] = raw.groupby(track_keys, sort=False)["Dapp_log"].transform("first")
    if max_area is not None:
        raw = raw.loc[raw["area"] < max_area].copy()
    selected = _pmk_feature_filter(raw, dapp_range)
    if selected.empty:
        raise ValueError("No localizations remain after area/Dapp_log filtering.")

    # Match the original transform's post-filter RangeIndex, but assign each
    # group's output back to its own rows instead of relying on group sorting.
    selected_source_rows = selected.index.to_numpy()
    selected = selected.reset_index(drop=True)
    pcs = np.empty((len(selected), 2), dtype=float)
    group_keys = track_keys if pc_groupby == "track" else cell_keys
    transform_columns = track_keys + ["fov_mask_id", "x_t_px", "y_t_px",
                                      "comment", "fov_uids"]
    transform_columns += [name for name in ("Dapp_log", "area") if name in selected.columns]
    for key, group in selected.groupby(group_keys, sort=False):
        group_rows = group.index.copy()
        group = group.loc[:, transform_columns].copy()
        object.__setattr__(group, "name", key)
        result = pc_transform(group, pixel=pixel_um, absPC=False)
        if isinstance(result, pd.DataFrame):
            if {"pc0", "pc1"}.issubset(result.columns):
                result = result.loc[:, ["pc0", "pc1"]]
            if (result.index.is_unique and len(result) == len(group_rows)
                    and group_rows.isin(result.index).all()):
                result = result.reindex(group_rows)
        values = np.asarray(result, dtype=float)
        if values.shape != (len(group_rows), 2):
            raise ValueError("to_PC_raw must return one (pc0, pc1) row per input row; "
                             f"received shape {values.shape} for {len(group_rows)} rows.")
        pcs[group_rows.to_numpy()] = values
    selected[["pc0", "pc1"]] = pcs
    selected.index = source_index.take(selected_source_rows)
    return selected


def _pmk_coordinates(df, fold_pc0):
    """Copy numeric coordinates without modifying the caller's DataFrame."""
    missing = [column for column in ("pc0", "pc1") if column not in df.columns]
    if missing:
        raise ValueError("Missing PC-coordinate columns: " + ", ".join(missing)
                         + ". Call prepare_pc_coordinates for raw input first.")
    coords = df.loc[:, ["pc0", "pc1"]].astype(float).copy()
    if fold_pc0:
        coords["pc0"] = -np.abs(coords["pc0"])
    for column in ("pc0", "pc1"):
        values = coords[column].to_numpy()
        values = values[np.isfinite(values)]
        if values.size < 2 or np.var(values, ddof=1) <= 0:
            raise ValueError(f"{column} needs at least two distinct finite values for KDE.")
    if not np.any(np.isfinite(coords.to_numpy()).all(axis=1)):
        raise ValueError("No finite (pc0, pc1) pairs are available for a heatmap.")
    return coords


def _pmk_edges(edges, name):
    """Require the equally spaced bin edges used by a rectangular pixel map."""
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError(f"{name} must be a finite, strictly increasing 1D array.")
    if not np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-7, atol=1e-12):
        raise ValueError(f"{name} must use equal-width bins for a pixel-mean reference.")
    return edges


def make_pc_heatmap_grid(df, *, heatmap_bins=15, fold_pc0=True):
    """Return (xedges, yedges) for a grid matching the original grid convention.

    The pc1 extent is symmetric about zero with 10 percent padding. The pc0
    bin width matches the pc1 bin width. With fold_pc0=True, the right pc0 edge
    is zero, matching get_heatmap_and_contour_mask's negative-half-axis view.
    With fold_pc0=False, the grid covers the supplied signed pc0 coordinates.
    Use a reference group's edges for all groups to share the same grid.
    This grid helper requires PC coordinates; use prepare_pc_coordinates first
    for raw input. The plotting entry points handle that conversion for you.
    """
    if (isinstance(heatmap_bins, bool) or int(heatmap_bins) != heatmap_bins
            or heatmap_bins < 2):
        raise ValueError("heatmap_bins must be an integer of at least 2.")
    coords = _pmk_coordinates(df, fold_pc0)
    xy = coords.to_numpy()
    xy = xy[np.isfinite(xy).all(axis=1)]
    yextent = 1.1 * np.max(np.abs(xy[:, 1]))
    if yextent <= 0:
        raise ValueError("Finite coordinate pairs need a nonzero pc1 extent to build the grid.")
    yedges = np.linspace(-yextent, yextent, int(heatmap_bins) + 1)
    spacing = yedges[1] - yedges[0]
    left_index = int(np.floor(xy[:, 0].min() / spacing))
    right_index = 0 if fold_pc0 else int(np.ceil(xy[:, 0].max() / spacing))
    right_index = max(right_index, left_index + 1)
    xedges = np.arange(left_index, right_index + 1, dtype=float) * spacing
    return xedges, yedges


def spherocylinder_marginal_pdf(coordinates, *, axis, half_length, radius):
    """Return the unsmoothed 1D density of a uniform 3D spherocylinder.

    The cell is centered at zero, with its cylinder along pc0. half_length is
    the distance from the center to either pole, INCLUDING a hemispherical cap.
    radius is the radius of the cylinder and both hemispheres. Both must use
    the SAME spatial scale as the plotted PC coordinates. This assumes equal
    detection probability throughout the cell volume, projected along depth.

    With cylinder length L = 2 * (half_length - radius) and cell volume V:
        pc0: density = circular cross-sectional area / V.
        pc1: density = [2*L*sqrt(radius**2-y**2)
                        + pi*(radius**2-y**2)] / V.
    The second expression includes BOTH the cylinder and the two caps.
    """
    if axis not in ("pc0", "pc1"):
        raise ValueError("axis must be 'pc0' or 'pc1'.")
    half_length, radius = float(half_length), float(radius)
    if (not np.isfinite([half_length, radius]).all()
            or radius <= 0 or half_length < radius):
        raise ValueError("Require finite half_length >= radius > 0, in the same PC units.")
    values = np.asarray(coordinates, dtype=float)
    half_cylinder = half_length - radius
    cylinder_length = 2 * half_cylinder
    volume = np.pi * radius**2 * cylinder_length + 4 * np.pi * radius**3 / 3
    if axis == "pc0":
        cap_distance = np.maximum(np.abs(values) - half_cylinder, 0.0)
        cross_section = np.pi * np.maximum(radius**2 - cap_distance**2, 0.0)
    else:
        squared_radius = np.maximum(radius**2 - values**2, 0.0)
        cross_section = (2 * cylinder_length * np.sqrt(squared_radius)
                         + np.pi * squared_radius)
    return cross_section / volume


def _pmk_resolve_cell_geometry(coords, half_length, radius):
    """Validate supplied geometry, or label an observed-extent approximation."""
    if (half_length is None) != (radius is None):
        raise ValueError("Provide both cell_half_length and cell_radius, or neither.")
    if half_length is None:
        half_length = np.max(np.abs(coords.loc[np.isfinite(coords.pc0), "pc0"]))
        radius = np.max(np.abs(coords.loc[np.isfinite(coords.pc1), "pc1"]))
        warnings.warn(
            "Cell geometry was approximated from observed PC-coordinate extents. "
            "These are not measured cell boundaries. For quantitative comparisons, "
            "set cell_half_length and cell_radius from cell geometry in the SAME "
            "PC-coordinate scale; do not use micrometers with normalized PCs.",
            UserWarning, stacklevel=3,
        )
    # Use the public density's geometry validation for both input routes.
    spherocylinder_marginal_pdf([0.0], axis="pc0", half_length=half_length, radius=radius)
    return float(half_length), float(radius)


def _pmk_spherocylinder_curve(samples, dim, half_length, radius, bw_adjust, grid=None):
    """Convolve the ideal marginal with the original KDE's Gaussian kernel.

    Deterministic Gaussian quadrature avoids drawing artificial localizations.
    The transverse sine substitution resolves the circular-edge behavior.
    """
    fitted = stats.gaussian_kde(samples, bw_method="scott")
    bandwidth = float(np.sqrt(fitted.covariance[0, 0])) * bw_adjust
    extent = half_length if dim == "pc0" else radius
    if grid is None:
        grid = np.linspace(-extent - 5 * bandwidth, extent + 5 * bandwidth, 1201)
    else:
        grid = np.asarray(grid, dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(192)
    if dim == "pc0":
        a = half_length - radius
        bounds = [(-half_length, -a), (-a, a), (a, half_length)]
        positions, masses = [], []
        for left, right in bounds:
            if right <= left:
                continue
            x = (right - left) * nodes / 2 + (right + left) / 2
            quadrature_weights = weights * (right - left) / 2
            pdf = spherocylinder_marginal_pdf(
                x, axis=dim, half_length=half_length, radius=radius
            )
            positions.append(x)
            masses.append(quadrature_weights * pdf)
        positions = np.concatenate(positions)
        masses = np.concatenate(masses)
    else:
        theta = nodes * np.pi / 2
        positions = radius * np.sin(theta)
        quadrature_weights = weights * np.pi / 2 * radius * np.cos(theta)
        masses = quadrature_weights * spherocylinder_marginal_pdf(
            positions, axis=dim, half_length=half_length, radius=radius
        )
    masses /= masses.sum()
    kernels = stats.norm.pdf((grid[:, None] - positions[None, :]) / bandwidth)
    density = kernels @ masses / bandwidth
    return grid, density


def circle_chord_pdf(coordinates, *, radius, center=0.0):
    """Normalized circular chord-length profile, before Gaussian smoothing.

    The geometric chord length at y is 2*sqrt(radius**2-(y-center)**2).
    Dividing by pi*radius**2 gives the marginal for uniform area in a disk,
    equivalently the transverse marginal of a uniformly populated cylinder.
    It does not include the additional contribution of hemispherical caps.
    """
    radius, center = float(radius), float(center)
    if not np.isfinite([radius, center]).all() or radius <= 0:
        raise ValueError("Require a finite positive radius and a finite center.")
    offset = np.asarray(coordinates, dtype=float) - center
    return 2 * np.sqrt(np.maximum(radius**2 - offset**2, 0.0)) / (np.pi * radius**2)


def _pmk_pc1_interval(interval, radius):
    """Resolve the PC1 reference interval independently of the data histogram."""
    if interval is None:
        return -radius, radius
    values = np.asarray(interval, dtype=float)
    if (values.shape != (2,) or not np.isfinite(values).all()
            or values[0] >= values[1]):
        raise ValueError("pc1_mean_range must be a finite (lower, upper) pair with lower < upper.")
    return float(values[0]), float(values[1])


def _pmk_adjustable_geometry_curve(
    samples, dim, half_length, radius, bw_adjust, *, hist_bins,
    pc0_mean_height, pc1_mean_range, pc1_profile, mean_smoothing,
):
    """Build a reference independently from the original plotted observations.

    PC0 is a shape over the fixed interval [-1, 1]. By default its negative
    half is normalized to area 1 AFTER optional smoothing and cropping to
    this interval. The earlier histogram-height choices remain available
    as explicit alternatives and are not area-normalized.

    PC1 uses a unit-area theoretical marginal stretched onto its selected
    interval. Gaussian smoothing is optional. Only the selected interval is
    drawn, so Gaussian tails outside it are not displayed or renormalized.
    """
    if dim == "pc0":
        grid = np.linspace(-1.0, 1.0, 1201)
        # Normalize the geometry by the center-to-pole distance, retaining
        # its aspect ratio while fixing the reference domain to [-1, 1].
        normalized_radius = radius / half_length
        if mean_smoothing:
            _, shape = _pmk_spherocylinder_curve(
                samples, "pc0", 1.0, normalized_radius, bw_adjust, grid=grid
            )
        else:
            shape = spherocylinder_marginal_pdf(
                grid, axis="pc0", half_length=1.0, radius=normalized_radius
            )
        if pc0_mean_height == "negative_area":
            # Remove numerical asymmetry only; no observations are mirrored.
            shape = (shape + shape[::-1]) / 2.0
            negative = grid <= 0.0  # Include zero as the integration endpoint.
            negative_area = trapezoid(shape[negative], grid[negative])
            if not np.isfinite(negative_area) or negative_area <= 0:
                raise ValueError("PC0 reference must have positive finite negative-side area.")
            # Normalize the displayed curve, not the untruncated Gaussian.
            # This preserves area 1 per half for either smoothing setting.
            return grid, shape / negative_area
        heights, _ = np.histogram(samples, bins=int(hist_bins), density=True)
        target_height = float(heights.mean())
        if pc0_mean_height == "half_hist_mean":
            target_height /= 2.0
        # Use the central height, not area, to preserve the requested switch.
        shape = shape / shape[len(grid) // 2]
        return grid, target_height * shape

    lower, upper = _pmk_pc1_interval(pc1_mean_range, radius)
    center = (lower + upper) / 2
    display_radius = (upper - lower) / 2
    grid = np.linspace(lower, upper, 1201)
    if pc1_profile == "spherocylinder":
        # Stretch only the PC1 reference, retaining the cell aspect ratio.
        # This parameter never changes PC0 or either original distribution.
        display_half_length = half_length / radius * display_radius
        if mean_smoothing:
            _, density = _pmk_spherocylinder_curve(
                samples, "pc1", display_half_length, display_radius, bw_adjust,
                grid=grid - center,
            )
        else:
            density = spherocylinder_marginal_pdf(
                grid - center, axis="pc1", half_length=display_half_length,
                radius=display_radius,
            )
    elif not mean_smoothing:
        density = circle_chord_pdf(grid, radius=display_radius, center=center)
    else:
        fitted = stats.gaussian_kde(samples, bw_method="scott")
        bandwidth = float(np.sqrt(fitted.covariance[0, 0])) * bw_adjust
        nodes, weights = np.polynomial.legendre.leggauss(192)
        theta = nodes * np.pi / 2
        positions = center + display_radius * np.sin(theta)
        # Chord PDF times dy, with y = center + R*sin(theta).
        masses = weights * np.cos(theta)**2
        masses /= masses.sum()
        kernels = stats.norm.pdf((grid[:, None] - positions[None, :]) / bandwidth)
        density = kernels @ masses / bandwidth
    return grid, density


def _pmk_reference_curve(samples, edges, heatmap, dim, mean_mode, bw_adjust):
    """Compute a reference using the original KDE's absolute bandwidth."""
    fitted = stats.gaussian_kde(samples, bw_method="scott")
    bandwidth = float(np.sqrt(fitted.covariance[0, 0])) * bw_adjust
    grid = np.linspace(edges[0] - 4 * bandwidth, edges[-1] + 4 * bandwidth, 1024)

    if mean_mode == "all_map":
        # Replacing every equal-area pixel by heatmap.mean() gives a uniform
        # marginal. The common mean cancels under probability normalization.
        # Integrate that uniform marginal against the Gaussian kernel exactly.
        density = (
            stats.norm.cdf((grid - edges[0]) / bandwidth)
            - stats.norm.cdf((grid - edges[-1]) / bandwidth)
        ) / (edges[-1] - edges[0])
    else:
        # imshow orientation: rows are pc1, columns are pc0. Include zeros.
        means = heatmap.mean(axis=0 if dim == "pc0" else 1)
        weights = means / means.sum()
        centers = (edges[:-1] + edges[1:]) / 2
        kernels = stats.norm.pdf((grid[:, None] - centers[None, :]) / bandwidth)
        density = (kernels @ weights) / bandwidth
    return grid, density


def plot_pc_mean_kde(
    df, *, mean_mode="spherocylinder", fold_pc0=True, heatmap_bins=15,
    hist_bins=30, bw_adjust=1.0, xedges=None, yedges=None, heatmap=None,
    title="", figsize=(12, 4.5), axes=None, save_path=None, show=True,
    dapp_range=None, pixel_um=0.106, max_area=2.5, pc_groupby="track",
    experiment_id=None, pc_transform=None, area_func=None,
    cell_half_length=None, cell_radius=None,
    pc0_mean_height="negative_area", pc1_mean_range=None,
    pc1_profile="circle_chord", mean_smoothing=False,
):
    """Plot pc0 and pc1, each with an original histogram and two KDE lines.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per localization, either original raw data or pc0/pc1 data.
        Raw input is converted by prepare_pc_coordinates using the original
        project helpers before the histogram and either KDE are calculated.
    dapp_range : (lower, upper), optional
        Exclusive Dapp_log bounds. For raw data the first value per track is
        used, and selection occurs before coordinate conversion.
    pixel_um, max_area, pc_groupby, experiment_id, pc_transform, area_func
        Raw-data options passed to prepare_pc_coordinates. Defaults match
        spatial_distribution: pixel=0.106, area < 2.5, PC groups include
        particle. These options do not recompute already-PC input.
    mean_mode : {'spherocylinder', 'all_map', 'by_axis'}
        'spherocylinder' (default): geometric reference shapes, with adjustable
        PC0 normalization and PC1 range. PC0 is always drawn over [-1, 1] and uses the
        cell's aspect ratio for its plateau and caps. PC1 uses pc1_profile.
        The original observations are processed separately, including the
        original negative-half-axis folding when fold_pc0=True.
        'all_map': average ALL pixels across the full rectangular
        map, including background zeros. The orange line is a uniform spatial
        reference smoothed with the original KDE bandwidth. No mask file is
        needed because the common mean count cancels after normalization.
        'by_axis': use column means for pc0 and row means for pc1 as KDE
        weights at bin centers. This usually resembles the original KDE.
    fold_pc0 : bool
        True (default) uses -abs(pc0) for the ORIGINAL histogram and KDE,
        matching get_heatmap_and_contour_mask. Already folded negative values
        remain unchanged. False plots the supplied PC0 values as stored,
        which is useful for the original ready_pc_heatmaps input convention.
        This option does not fold or mirror the geometric reference line.
    pc0_mean_height : {'negative_area', 'hist_mean', 'half_hist_mean'}
        'negative_area' (default) divides the reference by its numerical
        integral over [-1, 0], AFTER optional smoothing. The negative side
        has area 1, matching a density histogram of -abs(pc0). The curve is
        symmetric, so the positive side also has area 1 and the total is 2.
        It is a mirrored folded-density reference, not a full-domain PDF.
        This choice is independent of histogram bin heights and bin count.

        For the earlier height benchmarks, 'hist_mean' uses the arithmetic
        mean of all original density-histogram bin heights (including empty
        bins) as the central reference height; 'half_hist_mean' halves it.
        The original automatic bin range is retained. The mean is not the
        mean PC0 position or the mean KDE height. The curve is NOT area-
        normalized with these two legacy choices, so they differ by exactly
        a factor of two everywhere and do not guarantee unit negative area.
        The drawing interval is fixed at [-1, 1] for all three choices.
    pc1_mean_range : (lower, upper), optional
        Sets the PC1 reference's width and exact drawing interval. Its center
        is the interval midpoint; use (-R, R) to keep zero as the center.
        None uses (-cell_radius, cell_radius). This stretches the theoretical
        reference horizontally; it is not an axis-limit or data-filter option.
        The PC0 reference and both original distributions are unaffected.
    pc1_profile : {'circle_chord', 'spherocylinder'}
        'circle_chord' (default) uses 2*sqrt(R**2-y**2), normalized by pi*R**2;
        it is the projection of uniform area in a disk, equivalently the
        transverse marginal of a uniformly populated cylinder.
        'spherocylinder' includes the cylindrical body and both hemispherical
        caps of a uniformly populated 3D cell.
        These are related arched profiles, but not identical models.
    mean_smoothing : bool
        False (default) draws the unsmoothed theoretical shapes with zero
        endpoint values. True smooths them with the original KDE's Gaussian
        bandwidth in each panel. Only the requested intervals are drawn:
        Gaussian tails outside them are not displayed and values at the
        boundaries can be positive. PC0 'negative_area' normalization occurs
        after this truncation, retaining unit area on [-1, 0]. PC1 retains
        its existing normalization, so smoothing can reduce its displayed
        area below 1 by hiding tails. This does not change either histogram
        or original KDE.
    cell_half_length, cell_radius : float, optional
        Geometry for 'spherocylinder', with both quantities in a common scale.
        The half-length includes a cap. PC0 uses the ratio radius/half_length
        and a fixed normalized half-length of 1. The full-cell PC1 option
        retains that aspect ratio when stretched to pc1_mean_range.
        The circle-chord PC1 option depends only on its interval.
        Without an explicit PC1 range, the
        radius must also match the PC1 coordinate units.
        Supply both from cell geometry, not a fitted localization density.
        If both are omitted, max(abs(pc0)) and max(abs(pc1)) provide an
        approximate observed-extent model and a warning is emitted.
        A single ideal geometry is used for this dataset. Pooled cells with
        different shapes are not automatically modeled as a geometry mixture.
    heatmap_bins, hist_bins : int
        Separate controls: heatmap_bins sets the pc1 grid resolution (default
        15); hist_bins sets the original 1D histogram bin count (default 30).
    xedges, yedges : array-like, optional
        Provide both to reproduce a known heatmap grid or compare groups on
        one reference grid. Otherwise make_pc_heatmap_grid supplies the edges.
        These arguments and heatmap apply only to the two pixel modes.
    heatmap : 2D array, optional
        An existing count map in displayed orientation (rows=pc1, columns=pc0).
        Requires explicit xedges and yedges. Can be an already masked map.
        Otherwise an unmasked count map is constructed from finite coordinate
        pairs. Supplied edges may exclude points outside their range.
    axes : sequence of two matplotlib Axes, optional
        Allows embedding the two plots in an existing figure.
    save_path : str or pathlib.Path, optional
        Save the figure if provided; the parent folder is created as needed.

    Returns
    -------
    fig, axes
        The figure and a two-element array of axes for further customization.

    Notes
    -----
    The purple histogram and black KDE use the finite coordinates on each
    axis directly, not the heatmap, retaining the original PC0 folding rule.
    The orange line is not a KDE of intensity values or of an average position.
    The default PC0 reference integrates to 1 over [-1, 0] and 2 over [-1, 1].
    Its positive half is a symmetric display extension of the control group;
    it does not recover the unknown cell head. Ordinary Gaussian KDE of the
    folded observations can extend beyond [-1, 0], so the unchanged black
    KDE need not integrate to 1 on that interval or on the negative half-line.
    No original observation is duplicated or assigned a reconstructed sign.
    Geometry-height/range/profile controls apply only to spherocylinder mode.
    No existing plotting function is called, altered, or patched. Raw input
    calls only the original get_area and to_PC_raw preprocessing helpers.
    """
    if mean_mode not in ("spherocylinder", "all_map", "by_axis"):
        raise ValueError("mean_mode must be 'spherocylinder', 'all_map', or 'by_axis'.")
    if pc0_mean_height not in ("negative_area", "hist_mean", "half_hist_mean"):
        raise ValueError("pc0_mean_height must be 'negative_area', 'hist_mean', "
                         "or 'half_hist_mean'.")
    if pc1_profile not in ("spherocylinder", "circle_chord"):
        raise ValueError("pc1_profile must be 'spherocylinder' or 'circle_chord'.")
    if not isinstance(mean_smoothing, (bool, np.bool_)):
        raise TypeError("mean_smoothing must be True or False.")
    _pmk_pc1_interval(pc1_mean_range, 1.0)
    if not np.isfinite(bw_adjust) or bw_adjust <= 0:
        raise ValueError("bw_adjust must be positive and finite.")
    if isinstance(hist_bins, bool) or int(hist_bins) != hist_bins or hist_bins < 1:
        raise ValueError("hist_bins must be a positive integer.")
    prepared = prepare_pc_coordinates(
        df, dapp_range=dapp_range, pixel_um=pixel_um, max_area=max_area,
        pc_groupby=pc_groupby, experiment_id=experiment_id,
        pc_transform=pc_transform, area_func=area_func,
    )
    coords = _pmk_coordinates(prepared, fold_pc0)
    if mean_mode == "spherocylinder":
        if any(item is not None for item in (xedges, yedges, heatmap)):
            raise ValueError("Spherocylinder mode uses cell geometry; omit heatmap and bin edges.")
        cell_half_length, cell_radius = _pmk_resolve_cell_geometry(
            coords, cell_half_length, cell_radius
        )
        xedges = np.array([-1.0, 1.0])
        yedges = np.array(_pmk_pc1_interval(pc1_mean_range, cell_radius))
    else:
        if (xedges is None) != (yedges is None):
            raise ValueError("Provide both xedges and yedges, or neither.")
        if xedges is None:
            if heatmap is not None:
                raise ValueError("An existing heatmap requires its xedges and yedges.")
            xedges, yedges = make_pc_heatmap_grid(
                coords, heatmap_bins=heatmap_bins, fold_pc0=fold_pc0
            )
        xedges, yedges = _pmk_edges(xedges, "xedges"), _pmk_edges(yedges, "yedges")
        if heatmap is None:
            xy = coords.to_numpy()
            xy = xy[np.isfinite(xy).all(axis=1)]
            heatmap = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xedges, yedges])[0].T
        else:
            heatmap = np.asarray(heatmap, dtype=float)
        if heatmap.shape != (len(yedges) - 1, len(xedges) - 1):
            raise ValueError("heatmap must have shape (len(yedges)-1, len(xedges)-1).")
        if (not np.isfinite(heatmap).all() or np.any(heatmap < 0)
                or heatmap.sum() <= 0):
            raise ValueError("heatmap must have finite, nonnegative counts and a positive sum.")

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) != 2 or axes[0].figure is not axes[1].figure:
            raise ValueError("axes must contain two Axes from the same figure.")
        fig = axes[0].figure
    axes = np.asarray(axes, dtype=object).ravel()
    pixel_mean_label = {
        "spherocylinder": "Uniform-volume reference KDE",
        "all_map": "Mean pixel KDE (all map)",
        "by_axis": "Axis-mean KDE",
    }[mean_mode]

    for ax, dim, edges, label in zip(
        axes, ("pc0", "pc1"), (xedges, yedges),
        ("normalized length", "normalized width"),
    ):
        values = coords[dim].to_numpy()
        values = values[np.isfinite(values)]
        # >>> CHANGED >>>
        sns.histplot(x=values, bins=int(hist_bins), stat="density", color="#BDBDBD",
                     alpha=1.0, edgecolor="white", linewidth=0.5,
                     label="Original histogram", ax=ax)
        sns.kdeplot(x=values, color="#1F4E9C", lw=1.8, bw_adjust=bw_adjust,
                    label="Original KDE", ax=ax)
        # <<< CHANGED <<<
        #sns.histplot(x=values, bins=int(hist_bins), stat="density", color="slateblue",
                    # alpha=0.25, edgecolor=None, label="Original histogram", ax=ax)
        #sns.kdeplot(x=values, color="black", lw=1.8, bw_adjust=bw_adjust,
                    #label="Original KDE", ax=ax)
        if mean_mode == "spherocylinder":
            grid, density = _pmk_adjustable_geometry_curve(
                values, dim, cell_half_length, cell_radius, bw_adjust,
                hist_bins=hist_bins, pc0_mean_height=pc0_mean_height,
                pc1_mean_range=pc1_mean_range, pc1_profile=pc1_profile,
                mean_smoothing=mean_smoothing,
            )
            if dim == "pc0":
                detail = {
                    "negative_area": "negative area = 1",
                    "hist_mean": "hist mean",
                    "half_hist_mean": "hist mean / 2",
                }[pc0_mean_height]
            else:
                detail = "full cell" if pc1_profile == "spherocylinder" else "circle chord"
            mean_label = f"Mean {'KDE' if mean_smoothing else 'reference'} ({detail})"
        else:
            mean_label = pixel_mean_label
            grid, density = _pmk_reference_curve(
                values, edges, heatmap, dim, mean_mode, bw_adjust
            )
        ax.plot(grid, density, color="darkorange", linestyle="--", lw=2.0,
                label=mean_label)
        ax.set_xlabel(f"{dim}, {label}")
        ax.set_ylabel("Probability density")
        ax.set_title(f"{dim} - {label}")
        if mean_mode == "spherocylinder" or not fold_pc0:
            original_grid = ax.lines[-2].get_xdata()
            limit = 1.02 * max(np.abs(grid).max(), np.abs(original_grid).max(),
                               np.abs(values).max())
           # ax.set_xlim(-limit, limit) #activate this line 
            if dim == "pc1":
                ax.set_xlim(-1, 1)          # width panel: fixed from -1 to 1
            else:
                ax.set_xlim(-1.5, 0.0)  
        handles, labels = ax.get_legend_handles_labels()
        names = ("Original histogram", "Original KDE", mean_label)
        selected = [labels.index(name) for name in names]
        ax.legend([handles[i] for i in selected], names, frameon=False,
                  loc="upper right", fontsize=8.5)
        if dim == "pc0":
            ax.set_ylim(0, 2.0)              # length panel: fixed y axis 0-2.2
        elif dim == "pc1":
            ax.set_ylim(0, 2.0)                # width panel: fixed y axis 0-2
        elif mean_mode == "spherocylinder":
            # Keep the legend above the data instead of covering either curve.
            ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        else:
            ax.set_ylim(bottom=0)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_pc_mean_kde_from_csv(
    folder, coord_paths, dapp_ranges=None, *, mean_mode="spherocylinder", fold_pc0=True,
    heatmap_bins=15, hist_bins=30, bw_adjust=1.0, shared_grid=True,
    output_dir=None, show=True,
    pixel_um=0.106, max_area=2.5, pc_groupby="track",
    pc_transform=None, area_func=None,
    cell_half_length=None, cell_radius=None,
    pc0_mean_height="negative_area", pc1_mean_range=None,
    pc1_profile="circle_chord", mean_smoothing=False,
):
    """Read original raw CSVs or exported PC-coordinate CSVs and plot them.

    coord_paths is a sequence of CSV filenames, relative to folder (absolute
    paths also work). Raw files are passed through prepare_pc_coordinates,
    reusing the original to_PC_raw and get_area. PC files bypass conversion.
    dapp_ranges is None, or a sequence of None/(lower, upper), one per file.
    Filtering uses lower < Dapp_log < upper, as in ready_pc_heatmaps. Use the
    same limits as the original analysis; None means no additional filtering.
    For raw files, selection uses the first Dapp_log value per track, after
    area selection and before PC conversion, as in spatial_distribution.
    Raw defaults: pixel_um=0.106, max_area=2.5, pc_groupby='track'. A missing
    experiment column is set to experiment0, experiment1, ... per input file.
    An existing experiment column is retained.

    With fold_pc0=True, the original PC0 histogram and KDE use -abs(pc0),
    including already-folded negative PC exports. The geometric PC0 reference
    is independent: it is drawn over [-1, 1]. By default pc0_mean_height is
    'negative_area', giving unit area over [-1, 0] and full area 2, even with
    smoothing enabled. 'hist_mean' and 'half_hist_mean' retain the earlier
    height benchmarks without guaranteeing unit area. pc1_mean_range sets
    only the PC1 reference range; pc1_profile defaults to 'circle_chord'.
    Use 'spherocylinder' for the full-cell volume marginal instead.
    mean_smoothing defaults to False and changes only the references.
    See plot_pc_mean_kde.
    The same supplied geometry and reference options apply to all files;
    without geometry each file gets an observed-extent approximation.

    shared_grid applies only to pixel modes: True uses the FIRST FILTERED file
    to set their heatmap grid. No heatmap grid is used for geometric references.
    Original histograms and black KDEs always use each file's own coordinates.
    Returns a list of (fig, axes), one figure per CSV, with two panels per figure.
    """
    if isinstance(coord_paths, (str, Path)):
        coord_paths = [coord_paths]
    paths = [Path(folder) / Path(path) for path in coord_paths]
    if not paths:
        raise ValueError("Provide at least one CSV file.")
    if dapp_ranges is None:
        dapp_ranges = [None] * len(paths)
    if len(dapp_ranges) != len(paths):
        raise ValueError("dapp_ranges must contain one entry per CSV file.")
    frames = []
    for i, (path, limits) in enumerate(zip(paths, dapp_ranges)):
        df = pd.read_csv(path)
        experiment_id = None if "experiment" in df.columns else f"experiment{i}"
        df = prepare_pc_coordinates(
            df, dapp_range=limits, pixel_um=pixel_um, max_area=max_area,
            pc_groupby=pc_groupby, experiment_id=experiment_id,
            pc_transform=pc_transform, area_func=area_func,
        )
        _pmk_coordinates(df, fold_pc0)
        frames.append(df)
    xedges = yedges = None
    if shared_grid and mean_mode != "spherocylinder":
        xedges, yedges = make_pc_heatmap_grid(
            frames[0], heatmap_bins=heatmap_bins, fold_pc0=fold_pc0
        )
    figures = []
    for i, (path, df) in enumerate(zip(paths, frames)):
        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{i:02d}_{path.stem}_{mean_mode}.svg"
        figures.append(plot_pc_mean_kde(
            df, mean_mode=mean_mode, fold_pc0=fold_pc0, heatmap_bins=heatmap_bins,
            hist_bins=hist_bins, bw_adjust=bw_adjust, xedges=xedges, yedges=yedges,
            title=path.stem, save_path=save_path, show=show,
            cell_half_length=cell_half_length, cell_radius=cell_radius,
            pc0_mean_height=pc0_mean_height, pc1_mean_range=pc1_mean_range,
            pc1_profile=pc1_profile, mean_smoothing=mean_smoothing,
        ))
    return figures
