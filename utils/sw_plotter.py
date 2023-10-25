import sys

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import clear_output, display

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'


class ContourPlot:
    def __init__(self):
        self.data_rows = []
        self.title_rows = []
        self.showxax_rows = []
        self.showyax_rows = []
        self.cmap_rows = []

    def img_plot(self, ax, x, title=None, showxax=True, showyax=True, cmap=None):
        if cmap is None:
            cmap = 'viridis'
        im = ax.imshow(x.T, cmap=cmap)
        if showxax:
            ax.set_xlabel('Time step')
        else:
            ax.get_xaxis().set_visible(False)
        if showyax:
            ax.set_ylabel('Position index')
        else:
            ax.get_yaxis().set_visible(False)

        # Adjust the colorbar size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        if title is not None:
            ax.set_title(title)

    def add_data(self, data, title=None, showxax=True, showyax=True, cmap=None):
        self.data_rows.append(data)
        self.title_rows.append(title or [""] * len(data))
        self.showxax_rows.append(showxax if isinstance(showxax, (list, tuple)) else [showxax] * len(data))
        self.showyax_rows.append(showyax if isinstance(showyax, (list, tuple)) else [showyax] * len(data))
        self.cmap_rows.append(cmap or ['viridis'] * len(data))

    def plot_data(self, global_title=None):
        total_rows = len(self.data_rows)
        total_cols = len(self.data_rows[0]) if self.data_rows else 0

        fig = plt.figure(figsize=(7 * total_cols, 2 * total_rows))
        gs = GridSpec(total_rows, total_cols, figure=fig, wspace=0.3, hspace=0.5)

        for row_idx, (data_row, title_row, showxax_row, showyax_row, cmap_row) in enumerate(
                zip(self.data_rows, self.title_rows, self.showxax_rows, self.showyax_rows, self.cmap_rows)):
            for col_idx, x in enumerate(data_row):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                self.img_plot(ax, x, title=title_row[col_idx],
                              showxax=showxax_row[col_idx],
                              showyax=showyax_row[col_idx],
                              cmap=cmap_row[col_idx])

        if global_title:
            fig.suptitle(global_title, fontsize=12)
        # Update the figure in Jupyter
        if 'ipykernel' in sys.modules:
            clear_output(wait=True)
            display(fig)
            plt.close(fig)


class OverlayLinePlot:
    def __init__(self, tsteps):
        self.data_zeta = []  # Contains datasets for zeta.
        self.data_v = []  # Contains datasets for v.
        self.labels = []
        self.tsteps = tsteps

    def add_data(self, zeta_data, v_data, label):
        """Adds data for zeta and v."""
        self.data_zeta.append(zeta_data)
        self.data_v.append(v_data)
        self.labels.append(label)

    def plot_data(self, title=None):
        num_tsteps = len(self.tsteps)

        # Adjusting the figsize here
        fig, axs = plt.subplots(num_tsteps, 2, figsize=(10, 2 * num_tsteps))

        for i, t in enumerate(self.tsteps):
            # Zeta plot
            ax_left = axs[i, 0]
            for data, label in zip(self.data_zeta, self.labels):
                ax_left.plot(data[t], label=label)
            ax_left.set_ylabel(f't={t}', rotation=0, horizontalalignment='right')
            if i == 0:
                ax_left.set_title('$\zeta$')
            if i == num_tsteps - 1:
                ax_left.set_xlabel('time step')
            ax_left.legend()

            # v plot
            ax_right = axs[i, 1]
            for data, label in zip(self.data_v, self.labels):
                ax_right.plot(data[t], label=label)
            if i == 0:
                ax_right.set_title('$v$')
            if i == num_tsteps - 1:
                ax_right.set_xlabel('time step')

        if title:
            fig.suptitle(title, fontsize=16)
        # Update the figure in Jupyter
        # check if kernel is ipython
        if 'ipykernel' in sys.modules:
            clear_output(wait=True)
            display(fig)
            plt.close(fig)
