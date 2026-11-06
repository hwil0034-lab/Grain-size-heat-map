"""
Tower Hill grain-size heatmap vs AGE  (numpy + matplotlib only)
=============================================================
Depth-vs-grain-size heatmap of %Chan (volume % per size bin) from the
Microtrac FLEX export (saved as CSV), with the radiocarbon dates shown
alongside.

Each numbered block lists at the top which variables it NEEDS from the
blocks before it and which it MAKES, so you can change one block without
breaking the others.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ======================================================================
# SETTINGS -- edit these
# ======================================================================
INPUT_CSV = r"C:\Users\Admin\OneDrive - Monash University\Documents\Research Project\Data\Vladislav Tower hill grain size\Tower_Hill_grainsize_09_02_2026_csv.csv"

OUTPUT_FIGURE = r"C:\Users\Admin\OneDrive - Monash University\Documents\Research Project\Data\Vladislav Tower hill grain size\tower_hill_grainsize_heatmap_AGE.png"
OUTPUT_FIGURE_MODEL = r"C:\Users\Admin\OneDrive - Monash University\Documents\Research Project\Data\Vladislav Tower hill grain size\tower_hill_age_depth_model.png"

# If the same depth was measured more than once (re-runs, "-a"/"-b",
# or the "-L" batch repeating the "00-55" / "58-83" batches):
#   "latest"  -> keep the most recent measurement (AcqDate + AcqTime)
#   "first"   -> keep the oldest measurement
#   "average" -> average all the measurements for that depth
DUPLICATE_STRATEGY = "latest"

# Any jump between two neighbouring samples bigger than this (in cm)
# is treated as missing core and shown as a white "No data" block.
GAP_THRESHOLD_CM = 10


# ======================================================================
# BLOCK 1 -- READ THE CSV
# Makes: header (column names), table (every cell as text)
# ======================================================================
all_cells = np.loadtxt(INPUT_CSV, delimiter=",", dtype=str, encoding="utf-8-sig")

header = all_cells[0]      # first row = column names
table = all_cells[1:]      # everything else = one row per sample

print("Read", table.shape[0], "samples and", table.shape[1], "columns")


# ======================================================================
# BLOCK 2 -- PULL OUT THE COLUMNS WE NEED
# Needs:  header, table
# Makes:  sample_ids, acq_dates, acq_times   (n,)  text
#         grain_sizes_um                     (n_bins,)
#         pct_chan                           (n, n_bins)
# ======================================================================
size_cols = np.where(np.char.startswith(header, "Size:"))[0]
chan_cols = np.where(np.char.startswith(header, "PctChan:"))[0]

sample_ids = table[:, list(header).index("ID:1")]
acq_dates = table[:, list(header).index("AcqDate")]
acq_times = table[:, list(header).index("AcqTime")]

# Grain sizes are the same for every sample, so take them from row 1
grain_sizes_um = table[0, size_cols].astype(float)
pct_chan = table[:, chan_cols].astype(float)

# FLEX lists bins COARSE -> FINE. Flip to FINE -> COARSE (left to right).
grain_sizes_um = grain_sizes_um[::-1]
pct_chan = pct_chan[:, ::-1]

print("Found", len(grain_sizes_um), "grain-size bins,",
      grain_sizes_um.min(), "to", grain_sizes_um.max(), "um")


# ======================================================================
# BLOCK 3 -- DEPTH OF EACH SAMPLE (FROM ITS NAME) AND WHEN IT WAS RUN
# Needs:  sample_ids, acq_dates, acq_times
# Makes:  depth_m    (n,)  middle of each sample interval, in metres
#         acq_stamp  (n,)  date+time as one number, e.g. 202601301624
#
# "THill-1192-1193"        -> 11.925 m
# "THill-00-55-1400-1401"  -> 14.005 m
# "THill-1400-1401-L"      -> 14.005 m
# Depth is always the LAST TWO all-number pieces between the dashes.
# ======================================================================
n = len(sample_ids)
depth_m = np.zeros(n)
acq_stamp = np.zeros(n)

for i in range(n):
    # --- depth ---
    pieces = sample_ids[i].split("-")
    numbers = [p for p in pieces if p.strip().isdigit()]
    top_cm = float(numbers[-2])
    bottom_cm = float(numbers[-1])
    depth_m[i] = (top_cm + bottom_cm) / 2 / 100

    # --- acquisition date/time ---
    # AcqDate looks like "1/30/2026 0:00" (month/day/year)
    # AcqTime looks like "1/0/1900 16:24" (only the time part matters)
    month, day, year = acq_dates[i].split(" ")[0].split("/")
    hour, minute = acq_times[i].split(" ")[-1].split(":")[:2]
    acq_stamp[i] = (int(year) * 1e8 + int(month) * 1e6 + int(day) * 1e4
                    + int(hour) * 100 + int(minute))


# ======================================================================
# BLOCK 4 -- SORT BY DEPTH AND DEAL WITH DUPLICATE DEPTHS
# Needs:  depth_m, pct_chan, sample_ids, acq_stamp, DUPLICATE_STRATEGY
# Makes:  depth_m, pct_chan, sample_ids  (sorted, one row per depth)
# ======================================================================
n_before = len(depth_m)

unique_depths = np.unique(depth_m)          # sorted, no repeats
new_pct_chan = []
new_ids = []

for d in unique_depths:
    matches = np.where(depth_m == d)[0]     # row numbers at this depth

    if DUPLICATE_STRATEGY == "average":
        new_pct_chan.append(pct_chan[matches].mean(axis=0))
        new_ids.append(sample_ids[matches[0]])
        continue

    if DUPLICATE_STRATEGY == "first":
        keep = matches[np.argmin(acq_stamp[matches])]   # oldest
    else:  # "latest"
        keep = matches[np.argmax(acq_stamp[matches])]   # newest
    new_pct_chan.append(pct_chan[keep])
    new_ids.append(sample_ids[keep])

depth_m = unique_depths
pct_chan = np.array(new_pct_chan)
sample_ids = np.array(new_ids)

print(f"Kept {len(depth_m)} samples ({n_before - len(depth_m)} duplicates "
      f"resolved by '{DUPLICATE_STRATEGY}'), depth "
      f"{depth_m.min():.2f}-{depth_m.max():.2f} m")


# ======================================================================
# BLOCK 5 -- FIND GAPS (MISSING CORE)
# Needs:  depth_m, GAP_THRESHOLD_CM
# Makes:  no_data_zones -- list of (top_m, bottom_m) to white out
#         half_step_m   -- half the normal sample spacing
#
# To set a gap by hand instead, replace the list at the end, e.g.
#     no_data_zones = [(15.06, 15.58)]
# ======================================================================
spacing_m = np.diff(depth_m)
half_step_m = np.median(spacing_m) / 2

no_data_zones = []
for i in range(len(spacing_m)):
    if spacing_m[i] * 100 > GAP_THRESHOLD_CM:
        gap_top = depth_m[i] + half_step_m
        gap_bottom = depth_m[i + 1] - half_step_m
        no_data_zones.append((gap_top, gap_bottom))

print("Normal sample spacing:", round(np.median(spacing_m) * 100, 1), "cm")
for top, bottom in no_data_zones:
    print(f"  No data between {top:.2f} m and {bottom:.2f} m")


# ======================================================================
# BLOCK 6 -- BUILD THE GRID EDGES FOR THE HEATMAP
# Needs:  depth_m, pct_chan, grain_sizes_um, half_step_m, no_data_zones
# Makes:  depth_edges, size_edges, heat_values
#
# Each sample is drawn as a band that reaches halfway to its neighbours.
# Inside a gap, an all-NaN band is added so nothing gets stretched
# across the missing section (NaN is drawn as blank).
# ======================================================================
depth_edges = [depth_m[0] - half_step_m]
heat_rows = []

for i in range(len(depth_m)):
    heat_rows.append(pct_chan[i])

    if i == len(depth_m) - 1:                       # last sample
        depth_edges.append(depth_m[i] + half_step_m)
        break

    # Is there a gap between this sample and the next one?
    in_gap = False
    for top, bottom in no_data_zones:
        if depth_m[i] < top and depth_m[i + 1] > bottom:
            in_gap = True

    if in_gap:
        depth_edges.append(depth_m[i] + half_step_m)      # end of this band
        heat_rows.append(np.full(pct_chan.shape[1], np.nan))
        depth_edges.append(depth_m[i + 1] - half_step_m)  # start of next band
    else:
        depth_edges.append((depth_m[i] + depth_m[i + 1]) / 2)

depth_edges = np.array(depth_edges)
heat_values = np.array(heat_rows)

# Grain-size edges: halfway between bins on a LOG scale
log_sizes = np.log10(grain_sizes_um)
log_mid = (log_sizes[:-1] + log_sizes[1:]) / 2
log_edges = np.concatenate(([log_sizes[0] - (log_mid[0] - log_sizes[0])],
                            log_mid,
                            [log_sizes[-1] + (log_sizes[-1] - log_mid[-1])]))
size_edges = 10 ** log_edges


# ======================================================================
# BLOCK 7 -- MEDIAN AND MODE GRAIN SIZE FOR EACH SAMPLE
# Needs:  depth_m, pct_chan, grain_sizes_um, GAP_THRESHOLD_CM
# Makes:  median_um, mode_um  (n,)  one value per sample
#         line_depth, line_median, line_mode -- same, but with a NaN
#         put in at each gap so the plotted line breaks there
# ======================================================================
median_um = np.zeros(len(depth_m))
mode_um = np.zeros(len(depth_m))

for i in range(len(depth_m)):
    # Median (D50): size where the cumulative % (fine -> coarse) hits 50 %.
    # Interpolated on log(size) because the bins are spaced logarithmically.
    cumulative = np.cumsum(pct_chan[i])
    median_um[i] = 10 ** np.interp(50, cumulative, np.log10(grain_sizes_um))

    # Mode: the size bin holding the most volume
    mode_um[i] = grain_sizes_um[np.argmax(pct_chan[i])]

# Put NaN into the gaps so the line is not drawn across missing core
gap_after = np.where(np.diff(depth_m) * 100 > GAP_THRESHOLD_CM)[0] + 1
line_depth = np.insert(depth_m, gap_after, np.nan)
line_median = np.insert(median_um, gap_after, np.nan)
line_mode = np.insert(mode_um, gap_after, np.nan)

print(f"Median: {median_um.min():.1f}-{median_um.max():.1f} um, "
      f"Mode: {mode_um.min():.1f}-{mode_um.max():.1f} um")




# ======================================================================
# BLOCK 7b -- SMOOTHED MEDIAN AND MODE (running mean)
# Needs:  line_median, line_mode (with NaN at gaps)
# Makes:  smooth_median, smooth_mode -- same length as the lines
#
# SMOOTH_WINDOW = how many samples are averaged together. Samples are
# every 2 cm, so 11 samples = ~22 cm = ~400 years with the age model.
# Bigger number = smoother line. Use an odd number.
# Each side of a gap is smoothed separately, so the gap is never bridged.
# ======================================================================
SMOOTH_WINDOW = 11

def running_mean(values, window):
    """Running mean that restarts at every NaN (gap)."""
    result = np.full(len(values), np.nan)
    kernel = np.ones(window)
    has_data = ~np.isnan(values)
    start = 0
    while start < len(values):
        if not has_data[start]:
            start += 1
            continue
        end = start
        while end < len(values) and has_data[end]:
            end += 1
        # values[start:end] is one unbroken stretch of core
        stretch = values[start:end]
        total = np.convolve(stretch, kernel, mode="same")
        count = np.convolve(np.ones(len(stretch)), kernel, mode="same")
        result[start:end] = total / count   # count is smaller at the ends
        start = end
    return result

smooth_median = running_mean(line_median, SMOOTH_WINDOW)
smooth_mode = running_mean(line_mode, SMOOTH_WINDOW)


# ======================================================================
# BLOCK 8 -- RADIOCARBON DATES
# Standalone -- typed in from Recalibrated_ranges.xlsx (SHCal20).
# Edit or add values here. Depths in cm, ages in cal yr BP.
# ======================================================================
c14_depth_top_cm    = np.array([1138,  1240,  1430,  1504,  1543])
c14_depth_bottom_cm = np.array([1139,  1241,  1432,  1505,  1544])
c14_age_bp          = np.array([17355.5, 19542, 22519, 25410.5, 25107.5])  # SHCal20
c14_age_young_bp    = np.array([16805, 19167, 22036, 22041, 24490])        # lower bound
c14_age_old_bp      = np.array([17906, 19917, 23002, 28780, 25725])        # upper bound


# Which dates to USE (True) or leave out (False), in the same order as above.
# 1504-1505 cm is left out: its calibrated range (+/-1460 yr) is too wide.
c14_use = np.array([True, True, True, False, True])

c14_depth_top_cm    = c14_depth_top_cm[c14_use]
c14_depth_bottom_cm = c14_depth_bottom_cm[c14_use]
c14_age_bp          = c14_age_bp[c14_use]
c14_age_young_bp    = c14_age_young_bp[c14_use]
c14_age_old_bp      = c14_age_old_bp[c14_use]

c14_depth_m = (c14_depth_top_cm + c14_depth_bottom_cm) / 2 / 100
print("Using", c14_use.sum(), "of", len(c14_use), "radiocarbon dates")


# ======================================================================
# BLOCK 9 -- AGE-DEPTH MODEL (straight line, weighted by date error)
# Needs:  c14_depth_m, c14_age_bp, c14_age_young_bp, c14_age_old_bp
# Makes:  rate_yr_per_m, intercept_yr
#
# age (cal yr BP) = rate_yr_per_m * depth (m) + intercept_yr
# Dates with big error bars (e.g. 1504 cm, +/-1460 yr) count for less.
# ======================================================================
c14_sigma = (c14_age_old_bp - c14_age_young_bp) / 4     # 95% range -> ~1 sigma
rate_yr_per_m, intercept_yr = np.polyfit(c14_depth_m, c14_age_bp, 1,
                                         w=1 / c14_sigma)

residuals = c14_age_bp - (rate_yr_per_m * c14_depth_m + intercept_yr)

print(f"Age model: age = {rate_yr_per_m:.0f} x depth(m) + {intercept_yr:.0f}")
print(f"Sedimentation rate: {1000 / rate_yr_per_m:.2f} mm/yr")
print("Residuals (yr):", np.round(residuals).astype(int))


# ======================================================================
# BLOCK 10 -- CONVERT EVERYTHING FROM DEPTH TO AGE (in ka = 1000 yr)
# Needs:  rate_yr_per_m, intercept_yr, depth_edges, no_data_zones,
#         line_depth
# Makes:  age_edges_ka, no_data_zones_ka, line_age_ka
# ======================================================================
def depth_to_age_ka(depth):
    """Depth in m -> modelled age in cal ka BP."""
    return (rate_yr_per_m * np.asarray(depth) + intercept_yr) / 1000

age_edges_ka = depth_to_age_ka(depth_edges)
line_age_ka = depth_to_age_ka(line_depth)          # NaNs stay NaN
no_data_zones_ka = [(depth_to_age_ka(top), depth_to_age_ka(bottom))
                    for top, bottom in no_data_zones]

print(f"Heatmap covers {age_edges_ka[0]:.1f}-{age_edges_ka[-1]:.1f} cal ka BP")



# ======================================================================
# BLOCK 11 -- FIGURE 1: GRAIN-SIZE HEATMAP VS AGE
# Needs:  size_edges, age_edges_ka, heat_values, no_data_zones_ka,
#         line_age_ka, line_median, line_mode, smooth_median, smooth_mode
# ======================================================================
# Colours: white (0 %) -> violet -> blue -> cyan -> green -> yellow -> red
CMAP = LinearSegmentedColormap.from_list(
    "gs_cmap",
    ["white", "violet", "blue", "cyan", "green", "yellow", "orange", "red"],
)

# layout="constrained" makes matplotlib work out the spacing itself every
# time the figure is drawn (including when you resize the plot window),
# so the colour bar never lands on top of the "Grain size" title.
fig, ax = plt.subplots(figsize=(10, 11), layout="constrained")

# ---- 11a. Heatmap -----------------------------------------------------
mesh = ax.pcolormesh(size_edges, age_edges_ka, heat_values,
                     cmap=CMAP, vmin=0, vmax=np.nanmax(heat_values))

ax.set_xscale("log")
ax.set_xlim(size_edges[0], size_edges[-1])
ax.set_ylim(age_edges_ka[-1], age_edges_ka[0])     # older = further down
ax.set_ylabel("Modelled age (cal ka BP)")

# Horizontal age grid -- change GRID_STEP_KA for a coarser/finer grid
GRID_STEP_KA = 0.5
grid_ages = np.arange(np.ceil(age_edges_ka[0] / GRID_STEP_KA) * GRID_STEP_KA,
                      age_edges_ka[-1], GRID_STEP_KA)
ax.set_yticks(grid_ages)
ax.grid(axis="y", color="k", lw=0.5, alpha=0.3)

# Only label whole ka (15, 16, 17 ...); the half-ka lines stay unlabelled
ax.set_yticklabels([f"{a:.0f}" if a % 1 == 0 else "" for a in grid_ages])

# ---- 11b. Grain-size class boundaries + labels along the bottom -------
# Delete this section if you don't want it.
class_bounds = [0.2, 0.63, 2.0, 6.3, 20, 63, 200, 630]
class_labels = ["fC", "mC", "cC", "fSi", "mSi", "cSi", "fS", "mS", "cS"]
for b in class_bounds:
    ax.axvline(b, color="k", lw=0.5, ls="--", alpha=0.5)

ax.set_xticks(class_bounds)
ax.set_xticklabels([str(b) for b in class_bounds])

label_edges = [size_edges[0]] + class_bounds + [size_edges[-1]]
label_mids = [np.sqrt(label_edges[i] * label_edges[i + 1])
              for i in range(len(label_edges) - 1)]
ax_cls = ax.twiny()
ax_cls.set_xscale("log")
ax_cls.set_xlim(ax.get_xlim())
ax_cls.xaxis.set_ticks_position("bottom")
ax_cls.xaxis.set_label_position("bottom")
ax_cls.spines["bottom"].set_position(("outward", 22))
ax_cls.spines["bottom"].set_visible(False)
ax_cls.set_xticks(label_mids)
ax_cls.set_xticklabels(class_labels)
ax_cls.minorticks_off()
ax_cls.tick_params(length=0)
ax_cls.set_xlabel("Grain size (\u00b5m)")

# ---- 11c. Colour scale UNDER the figure (below the axis titles) -------
# The constrained layout (see plt.subplots above) places this below the
# "Grain size" title automatically. pad = extra gap between the two
# (as a fraction of the figure), shrink = bar length, aspect = thickness.
cbar = fig.colorbar(mesh, ax=ax, location="bottom",
                    pad=0.02, aspect=45, shrink=0.8)
cbar.set_label("Volume (%)")

# ---- 11d. White "No data" blocks over missing core --------------------
for top, bottom in no_data_zones_ka:
    ax.axhspan(top, bottom, facecolor="white", edgecolor="0.4",
               hatch="///", lw=0.8, zorder=5)
    ax.text(np.sqrt(size_edges[0] * size_edges[-1]), (top + bottom) / 2,
            "No data", ha="center", va="center", fontsize=11,
            fontweight="bold", color="0.25", zorder=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))

# ---- 11e. Median + mode on ONE shared axis, drawn over the heatmap ----
# [left, width] as fractions of the heatmap width -- change to move it.
PROFILE_POSITION = [0.015, 0.22]

MEDIAN_RAW = "#f4a3a3"     # light red
MEDIAN_MEAN = "#8b0000"    # dark red
MODE_RAW = "#9cc7ea"       # light blue
MODE_MEAN = "#08306b"      # dark blue

left, width = PROFILE_POSITION
p_ax = ax.inset_axes([left, 0, width, 1])          # full height of heatmap

p_ax.plot(line_median, line_age_ka, color=MEDIAN_RAW, lw=0.8, zorder=10,
          label="Median")
p_ax.plot(line_mode, line_age_ka, color=MODE_RAW, lw=0.8, zorder=10,
          label="Mode")
p_ax.plot(smooth_median, line_age_ka, color=MEDIAN_MEAN, lw=1.8, zorder=11,
          label=f"Median {SMOOTH_WINDOW}-pt mean")
p_ax.plot(smooth_mode, line_age_ka, color=MODE_MEAN, lw=1.8, zorder=11,
          label=f"Mode {SMOOTH_WINDOW}-pt mean")

p_ax.set_ylim(ax.get_ylim())
p_ax.set_xlim(0, np.nanmax(line_mode) * 1.05)
p_ax.set_facecolor("none")                          # see-through
for side in ("left", "right", "bottom"):
    p_ax.spines[side].set_visible(False)
p_ax.set_yticks([])
p_ax.xaxis.tick_top()
p_ax.xaxis.set_label_position("top")
p_ax.set_xlabel("Median / mode (\u00b5m)", fontsize=9)
p_ax.tick_params(axis="x", labelsize=8)
p_ax.locator_params(axis="x", nbins=4)

# Legend sits above the heatmap, just right of the profile's scale
p_ax.legend(loc="lower left", bbox_to_anchor=(1.08, 1.01), ncol=2,
            fontsize=7, frameon=False, handlelength=1.5,
            borderpad=0, borderaxespad=0, columnspacing=1.0)

fig.suptitle("Tower Hill NWC \u2013 grain-size distribution vs age (% per bin)")

plt.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.3)
print("Saved figure to", OUTPUT_FIGURE)


# ======================================================================
# BLOCK 12 -- FIGURE 2: AGE-DEPTH MODEL (separate figure)
# Needs:  c14_* arrays, rate_yr_per_m, intercept_yr, residuals,
#         depth_edges, no_data_zones, depth_to_age_ka
# ======================================================================
fig2, ax2 = plt.subplots(figsize=(6, 7))

# Missing core
for top, bottom in no_data_zones:
    ax2.axhspan(top, bottom, facecolor="white", edgecolor="0.5",
                hatch="///", lw=0.8, zorder=1, label="No data")

# Model line: solid between the dates, dashed where it is extrapolated
first_date, last_date = c14_depth_m.min(), c14_depth_m.max()
d_inside = np.array([first_date, last_date])
d_above = np.array([depth_edges[0], first_date])
d_below = np.array([last_date, depth_edges[-1]])
ax2.plot(depth_to_age_ka(d_inside), d_inside, color="tab:red", lw=1.5,
         zorder=3, label=f"Linear model ({1000 / rate_yr_per_m:.2f} mm/yr)")
ax2.plot(depth_to_age_ka(d_above), d_above, color="tab:red", lw=1.5,
         ls="--", zorder=3, label="Extrapolated")
ax2.plot(depth_to_age_ka(d_below), d_below, color="tab:red", lw=1.5,
         ls="--", zorder=3)

# Calibrated dates with their full calibrated range
c14_age_ka = c14_age_bp / 1000
ax2.errorbar(c14_age_ka, c14_depth_m,
             xerr=[(c14_age_bp - c14_age_young_bp) / 1000,
                   (c14_age_old_bp - c14_age_bp) / 1000],
             fmt="o", color="k", ecolor="0.3", capsize=3, ms=5, zorder=4,
             label="\u00b9\u2074C date (SHCal20, cal range)")

for i in range(len(c14_depth_m)):
    ax2.annotate(f"{c14_age_bp[i]:,.0f}", (c14_age_ka[i], c14_depth_m[i]),
                 xytext=(6, 6), textcoords="offset points", fontsize=8,
                 zorder=5, bbox=dict(facecolor="white", edgecolor="none",
                                     pad=0.5))

# Equation in the corner
ax2.text(0.03, 0.30,
         f"Age = {rate_yr_per_m:.0f} \u00d7 depth (m) {intercept_yr:+.0f}\n"
         f"Sedimentation rate = {1000 / rate_yr_per_m:.2f} mm/yr",
         transform=ax2.transAxes, fontsize=8, va="bottom",
         bbox=dict(facecolor="white", edgecolor="0.7"))

ax2.set_ylim(depth_edges[-1], depth_edges[0])      # depth increases down
ax2.set_xlabel("Age (cal ka BP)")
ax2.set_ylabel("Depth (m)")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper right", fontsize=8, framealpha=1)
ax2.set_title("Tower Hill NWC \u2013 age-depth model")

plt.savefig(OUTPUT_FIGURE_MODEL, dpi=200, bbox_inches="tight", pad_inches=0.3)
print("Saved figure to", OUTPUT_FIGURE_MODEL)

plt.show()