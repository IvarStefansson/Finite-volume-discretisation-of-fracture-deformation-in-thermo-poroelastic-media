"""
It is assumed that the subfolder "results/" contains:
(1) normal_displacement_jumps_thm.txt
(2) tangential_displacement_jumps_thm.txt
(3) time_steps_thm.txt
(4) iterations_thm.txt

The figure file displacement_jumps_and_iterations.pdf will be generated inside
the "figures/" subfolder.
"""
# %% Importing modules
import numpy as np
import itertools
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter


def identify_time_scale(t):

    if t[-1] < 1 / 10:
        label = "Millisecond"
        label = "ms"
        t = t * 1000
    else:
        label = "s"
    return t, label


matplotlib.use("agg", warn=False, force=True)  # force non-GUI backend.

plt.close()
# Folder and file names
output_folder = "figures/"
files = [
    "normal_displacement_jumps_",
    "tangential_displacement_jumps_",
    "time_steps_",
    "iterations_",
]
extension = "thm.txt"
for i, f in enumerate(files):
    files[i] = "results/" + f + extension  #


# Loading Data
u_normal = np.loadtxt(files[0], skiprows=0, delimiter=" ", encoding="utf-8")
u_tang = np.loadtxt(files[1], skiprows=0, delimiter=" ", encoding="utf-8")
times = np.loadtxt(files[2], skiprows=0, delimiter=" ", encoding="utf-8")
iterations = np.loadtxt(files[3], skiprows=0, delimiter=" ", encoding="utf-8")
# No iterations recorded for initial time
iterations = np.hstack((0, iterations))

# Only plot for t >= 0
ind = times > -1e-5
times = times[ind]
u_tang = u_tang[ind]
u_normal = u_normal[ind]
iterations = iterations[ind]
# Identify time scale
times, label = identify_time_scale(times)

# Number of floating points
mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
mf.set_powerlimits((-4, 4))

# %% Plotting

# Preparing plot
sns.set_context("paper")  # set scale and size of figures
sns.set_palette("tab10", 10)
itertools.cycle(sns.color_palette())  # iterate if > 10 colors are needed


n_frac = u_normal.shape[1]
# Plotting data
def plot_single(
    times, u_normal, u_tang, i, ax1, normal_color=None, tangential_color=None
):
    times, label = identify_time_scale(times)
    if normal_color is None:
        normal_color = "red"
    if tangential_color is None:
        tangential_color = "blue"
    ax1.semilogy(times, u_normal[:, i], "--", color=normal_color, linewidth=1)
    ax1.semilogy(times, u_tang[:, i], "-", color=tangential_color, linewidth=1)


iteration_color = "black"
iteration_marker = ":"


def add_iterations(ax1, times, iterations):
    times, label = identify_time_scale(times)
    ax = ax1.twinx()
    ax.set_ylim(0, 12)
    return ax


colors = ["red", "blue", "green", "orange", "magenta", "grey", "brown", "cyan"]

ind_p = times < 2.001e-2
times_p = times[ind_p]

iterations_p = iterations[ind_p]
u_normal_p = u_normal[ind_p]
u_tang_p = u_tang[ind_p]
fig = plt.figure(8, constrained_layout=False, figsize=(6, 4))
gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=0.40, top=0.78)
gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.40, right=1.00, top=0.78)
gs3 = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=1.00, bottom=0.80)

with sns.axes_style("white"):
    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs2[0, 0])
    ax3 = fig.add_subplot(gs3[0, 0])

for i in range(n_frac):
    plot_single(times_p, u_normal_p, u_tang_p, i, ax1, colors[i], colors[i])

times_p, label = identify_time_scale(times_p)
r = times_p[-1] - times_p[0]

ax1.set_xlim(times_p[0] - r / 20, times_p[-1] + r / 20)
ax1.set_ylim(5e-8, 5e-3)

ax1.set_xlabel(r"$t$ [ms]")
ax1.set_ylabel(r"[m]")


ax1.tick_params(axis="both", which="major")
ax1.xaxis.set_major_formatter(mf)
ax1.yaxis.set_major_formatter(FormatStrFormatter("%1.0e"))

ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax1_it = add_iterations(ax1, times_p, iterations_p)
ax1_it.set_yticks(np.array([]))
ax1_it.plot(times_p, iterations_p, ls=iteration_marker, color=iteration_color)

ind_T = ~ind_p
times_T = times[ind_T]
iterations_T = iterations[ind_T]
u_normal_T = u_normal[ind_T]
u_tang_T = u_tang[ind_T]

for i in range(n_frac):
    c = colors[i]
    plot_single(times_T, u_normal_T, u_tang_T, i, ax2, c, c)
    ax3.plot([], [], "-", color=c, label="Fracture {}".format(i + 1))
    ax3.legend(loc="center", frameon=False, ncol=4)
    ax3.axis("off")

r = times_T[-1] - times_T[0]
ax2.set_xlim(times_T[0] - r / 20, times_T[-1] + r / 20)
ax2.set_ylim(5e-8, 5e-3)
ax2.set_xlabel(r"$t$ [s]")
ax2.set_yticks([], [])
ax2.set_xticks(np.array([1, 2, 3, 4, 5]))
ax2.tick_params(axis="both", which="both")
ax2.xaxis.set_major_formatter(mf)

ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax2it = add_iterations(ax2, times_T, iterations_T)
ax2it.set_yticks(np.array([2, 4, 6, 8, 10]))
ax2it.plot(times_T, iterations_T, ls=iteration_marker, color=iteration_color)
ax2.minorticks_off()
ax2it.set_ylabel("iterations")
# Add iteration legend
ax3.plot([], [], iteration_marker, color=iteration_color, label="Iterations")
ax3.legend(loc="center", frameon=False, ncol=4)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
fig.savefig(
    output_folder + "displacement_jumps_and_iterations.pdf", bbox_inches="tight"
)
