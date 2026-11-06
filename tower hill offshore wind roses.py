
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


# ======================================================================
# SETTINGS -- edit these
# ======================================================================
INPUT_DAT = r"C:\Users\Admin\OneDrive - Monash University\Documents\Research Project\Data\Sergey Wind modelling figures\stat-TH-os.dat"

OUTPUT_FIGURE_ROSES = r"C:\Users\Admin\OneDrive - Monash University\Documents\Research Project\Data\Sergey Wind modelling figures\tower_hill_offshore_wind_roses.png"
OUTPUT_FIGURE_IW = r"C:\Users\Admin\OneDrive - Monash University\Documents\Research Project\Data\Sergey Wind modelling figures\tower_hill_offshore_wind_run.png"

# Names exactly as they appear in the "data" column of the .dat file,
# and the short labels to show on the figure (top to bottom)
PERIODS = ["LGM_(MPI-ESM1.2-LR)", "MH_(MPI-ESM1.2-LR)", "PD_(ERA5)"]
PERIOD_LABELS = ["LGM (21 ka)\nMPI-ESM1.2-LR", "Mid-Holocene (6 ka)\nMPI-ESM1.2-LR",
                 "Present day\nERA5 (1990\u20132019)"]

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_LABELS = ["Summer (DJF)", "Autumn (MAM)", "Winter (JJA)", "Spring (SON)"]

# Wind speed colour classes (m/s) for Figure 1
SPEED_BINS = np.arange(2, 10.5, 1)        # 2, 3, 4 ... 10
SPEED_CMAP = "viridis"

# Outline colours for the three periods in Figure 2
PERIOD_COLOURS = ["#1f4e9c", "#e08a1e", "#7a7a7a"]   # LGM blue, MH orange, PD grey


# ======================================================================
# BLOCK 1 -- READ THE .DAT FILE
# Makes:  header (column names), table (every cell as text)
# The file is tab-separated: site, data, seas, par, unit, N NE E SE S SW W NW
# ======================================================================
all_cells = np.loadtxt(INPUT_DAT, delimiter="\t", dtype=str)

header = all_cells[0]
table = all_cells[1:]

DIRECTIONS = list(header[5:])              # ['N','NE','E','SE','S','SW','W','NW']
print("Read", table.shape[0], "rows; directions:", DIRECTIONS)


# ======================================================================
# BLOCK 2 -- A SMALL HELPER TO PULL OUT ONE ROW OF NUMBERS
# Needs:  table
# get_values("PD_(ERA5)", "JJA", "AWS") -> array of 8 numbers (N ... NW)
# ======================================================================
def get_values(period, season, parameter):
    """The 8 directional values for one period, season and statistic."""
    row = table[(table[:, 1] == period) &
                (table[:, 2] == season) &
                (table[:, 3] == parameter)]
    if len(row) != 1:
        raise ValueError(f"Expected 1 row for {period} {season} {parameter}, found {len(row)}")
    return row[0, 5:].astype(float)


# ======================================================================
# BLOCK 3 -- SUMMARY NUMBERS FOR THE DISCUSSION (printed, not plotted)
# Needs:  get_values, PERIODS, SEASONS, DIRECTIONS
#
#   dominant   = direction the wind most often blows FROM (highest RWO)
#   mean speed = average speed over all directions, weighted by how often
#                each direction occurs: sum(RWO * AWS) / 100
#   wind run   = total integrated wind (IW) for that season, in km
# ======================================================================
print("\nperiod                season  dominant (RWO %)   mean speed  wind run")
for period in PERIODS:
    for season in SEASONS + ["ANN"]:
        rwo = get_values(period, season, "RWO")
        aws = get_values(period, season, "AWS")
        iw = get_values(period, season, "IW")
        top = np.argmax(rwo)
        mean_speed = np.sum(rwo * aws) / np.sum(rwo)
        print(f"{period:22s} {season:5s}   {DIRECTIONS[top]:>3s} ({rwo[top]:4.1f}%)"
              f"        {mean_speed:4.1f} m/s   {iw.sum() / 1000:6.0f} km")


# ======================================================================
# BLOCK 4 -- ANGLES FOR THE 8 DIRECTIONS
# Makes:  angles (radians), bar_width
# Wind roses use "direction the wind blows FROM", N at the top, clockwise.
# ======================================================================
angles = np.deg2rad(np.arange(0, 360, 360 / len(DIRECTIONS)))   # N=0, NE=45 ...
bar_width = 2 * np.pi / len(DIRECTIONS) * 0.85                   # small gap between bars


def style_rose(ax, r_max, r_ticks, tick_suffix):
    """Common look for every rose: N at top, clockwise, same radial scale."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)                 # clockwise
    ax.set_xticks(angles)
    ax.set_xticklabels(DIRECTIONS, fontsize=7)
    ax.tick_params(axis="x", pad=-2)
    ax.set_ylim(0, r_max)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{t:g}{tick_suffix}" for t in r_ticks], fontsize=6,
                       color="0.4")
    ax.set_rlabel_position(112.5)              # put % labels between ESE and SE
    ax.grid(color="0.8", lw=0.5)
    ax.spines["polar"].set_color("0.6")


# ======================================================================
# BLOCK 5 -- FIGURE 1: 3 x 4 GRID, LENGTH = FREQUENCY, COLOUR = SPEED
# Needs:  get_values, angles, bar_width, style_rose, settings
# ======================================================================
norm = BoundaryNorm(SPEED_BINS, plt.get_cmap(SPEED_CMAP).N, extend="both")
cmap = plt.get_cmap(SPEED_CMAP)

# Same radial scale on every rose so they can be compared directly
max_rwo = max(get_values(p, s, "RWO").max() for p in PERIODS for s in SEASONS)
R_MAX = np.ceil(max_rwo / 10) * 10          # e.g. 38.2 % -> 40 %
R_TICKS = np.arange(10, R_MAX + 1, 10)

fig, axes = plt.subplots(len(PERIODS), len(SEASONS), figsize=(11, 9.5),
                         subplot_kw={"projection": "polar"}, layout="constrained")

for row, period in enumerate(PERIODS):
    for col, season in enumerate(SEASONS):
        ax = axes[row, col]
        rwo = get_values(period, season, "RWO")
        aws = get_values(period, season, "AWS")

        ax.bar(angles, rwo, width=bar_width, color=cmap(norm(aws)),
               edgecolor="0.2", linewidth=0.5, zorder=3)
        style_rose(ax, R_MAX, R_TICKS, "%")

        # Small summary under each rose: weighted mean speed
        mean_speed = np.sum(rwo * aws) / np.sum(rwo)
        ax.set_xlabel(f"mean {mean_speed:.1f} m/s", fontsize=7, labelpad=2)

        if row == 0:
            ax.set_title(SEASON_LABELS[col], fontsize=11, fontweight="bold", pad=14)
        if col == 0:
            ax.text(-0.42, 0.5, PERIOD_LABELS[row], transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=10,
                    fontweight="bold")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, location="bottom", shrink=0.5, aspect=35,
                    pad=0.02)
cbar.set_label("Average wind speed from each direction (m/s)")

fig.suptitle("Tower Hill offshore seasonal wind regimes \u2013 bar length = frequency of "
             "wind direction (%), colour = average speed", fontsize=11)

plt.savefig(OUTPUT_FIGURE_ROSES, dpi=200, bbox_inches="tight", pad_inches=0.2)
print("\nSaved figure to", OUTPUT_FIGURE_ROSES)


# ======================================================================
# BLOCK 6 -- FIGURE 2: WIND RUN (IW) OF THE THREE PERIODS, PER SEASON
# Needs:  get_values, angles, style_rose, settings
#
# Outlines are closed by repeating the first point at the end.
# IW is supplied in metres; shown here in km. IW is proportional to
# frequency x average speed, so compare it between periods and seasons
# (relative), rather than reading it as an absolute distance.
# ======================================================================
max_iw = max(get_values(p, s, "IW").max() for p in PERIODS for s in SEASONS) / 1000
IW_MAX = np.ceil(max_iw / 20) * 20          # e.g. 73.6 km -> 80 km
IW_TICKS = np.arange(20, IW_MAX + 1, 20)

fig2, axes2 = plt.subplots(1, len(SEASONS), figsize=(13, 4.2),
                           subplot_kw={"projection": "polar"}, layout="constrained")

closed_angles = np.append(angles, angles[0])

for col, season in enumerate(SEASONS):
    ax = axes2[col]
    for period, label, colour in zip(PERIODS, PERIOD_LABELS, PERIOD_COLOURS):
        iw = get_values(period, season, "IW") / 1000         # km
        closed_iw = np.append(iw, iw[0])
        ax.plot(closed_angles, closed_iw, color=colour, lw=2, zorder=4,
                label=label.split("\n")[0])
        ax.fill(closed_angles, closed_iw, color=colour, alpha=0.12, zorder=3)
    style_rose(ax, IW_MAX, IW_TICKS, " km")
    ax.set_title(SEASON_LABELS[col], fontsize=11, fontweight="bold", pad=14)

axes2[-1].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=8,
                 frameon=False)
fig2.suptitle("Tower Hill offshore integrated wind (wind run) by direction \u2013 "
              "frequency \u00d7 speed, a proxy for wind transport potential",
              fontsize=11)

plt.savefig(OUTPUT_FIGURE_IW, dpi=200, bbox_inches="tight", pad_inches=0.2)
print("Saved figure to", OUTPUT_FIGURE_IW)

plt.show()
