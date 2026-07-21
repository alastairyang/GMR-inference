import matplotlib.pyplot as plt
import numpy as np
import torch
from rasterio.transform import xy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import scipy.io as sio
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

class Plotting:
    def __init__(self):
        self.epsg     = 3031 # antarctic polar stereographic
        self.proj     = ccrs.SouthPolarStereo()
        self.data_crs = ccrs.epsg(self.epsg)
        pass

    def load_model(self, md):
        """  
        load our bayesian model for plotting
        """
        self.model = md
        return 
    
    def define_extent(self, x, y):
        self.x_min, self.x_max = np.min(x), np.max(x)
        self.y_min, self.y_max = np.min(y), np.max(y)
        self.x = x
        self.y = y
        return 
    
    def make_consistent(self, data_dict):
        """
        Check coordinate. If not consistent, interpolate

        data_dict: dict with keys 'x', 'y', 'data'
        """
        x, y = data_dict['x'], data_dict['y']
        if not (np.array_equal(x, self.x) and np.array_equal(y, self.y)):
            # interpolate to the common grid
            print("Interpolating data to the common grid...")
            interpolator = RegularGridInterpolator((y, x), data_dict['data'])
            X, Y = np.meshgrid(self.x, self.y)
            points = np.array([Y.flatten(), X.flatten()]).T
            data_dict['data'] = interpolator(points).reshape(len(self.y), len(self.x))
            data_dict['x'] = self.x
            data_dict['y'] = self.y
        return data_dict

    def plot(self, data, background, layout='row'):
        """
        Create (multipanel) plot.

        Parameters
        ----------
        data       : tuple of dict  — panels to plot (1, 2, or 3 entries)
        background : dict           — hillshade shown in all subplots
        layout     : str
            'row'        — original single-row layout (default)
            'main+stack' — one large panel left, two smaller stacked right
                        data[0] → large panel
                        data[1] → top-right small panel
                        data[2] → bottom-right small panel

        Dict keys for each panel:
            'x', 'y', 'data', 'vmin', 'vmax',
            'contour_data', 'contour_levels', 'contour_colors',
            'title', 'cb_label', 'cmap'
            (set unused keys to None)
        """
        if layout == 'main+stack':
            return self._plot_main_stack(data, background)
        else:
            return self._plot_row(data, background)

    # original row layout
    def _plot_row(self, data, background):
        n_panels = len(data)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(5 * n_panels, 6),
            subplot_kw={'projection': self.proj}
        )
        if n_panels == 1:
            axes = [axes]

        for ax, d in zip(axes, data):
            d = self.make_consistent(d)
            self._setup_ax(ax, background)
            self._add_scalebar(ax, length_km=200, n_segments=2)
            self._plot_panel(fig, ax, d)

        plt.tight_layout()
        return fig

    # private: main + two stacked layout
    def _plot_main_stack(self, data, background):
        assert len(data) == 3, "'main+stack' layout requires exactly 3 panels in data."

        fig = plt.figure(figsize=(13, 8))

        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[2, 1],   # left column twice as wide as right
            height_ratios=[1, 1],  # two equal rows
            wspace=0.08,
            hspace=0.15
        )

        # --- Axes ---
        ax_main   = fig.add_subplot(gs[0:2, 0], projection=self.proj)  # spans both rows
        ax_top    = fig.add_subplot(gs[0,   1], projection=self.proj)
        ax_bottom = fig.add_subplot(gs[1,   1], projection=self.proj)

        panels = [
            (ax_main,   data[0], data[0].get('scalebar')[0], data[0].get('scalebar')[1]),
            (ax_top,    data[1], data[1].get('scalebar')[0], data[1].get('scalebar')[1]),
            (ax_bottom, data[2], data[2].get('scalebar')[0], data[2].get('scalebar')[1]),
        ]

        for ax, d, sb_km, sb_seg in panels:
            d = self.make_consistent(d)
            self._setup_ax(ax, background)
            self._add_scalebar(ax, length_km=sb_km, n_segments=sb_seg)
            self._plot_panel(fig, ax, d)

        self._add_background_colorbar(fig, ax_main, background)

        ax_main.set_label('main')
        ax_top.set_label('top')
        ax_bottom.set_label('bottom')

        return fig

    # private: render a single panel (imshow + colorbar + contour)
    def _plot_panel(self, fig, ax, d):
        im = ax.imshow(
            d['data'],
            origin='lower',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            transform=self.data_crs,
            cmap=d.get('cmap', 'viridis'),
            alpha=1, zorder=1,
            vmin=d.get('vmin'), vmax=d.get('vmax')
        )
        ax.set_title(d.get('title'), fontsize=10)
        ax.set_aspect('equal')

        if d.get('colorbar', 'on') != 'off':
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        label=d.get('cb_label', ''))

        if d.get('contour_data') is not None:
            x, y = np.meshgrid(d['x'], d['y'])
            cs = ax.contour(
                x, y, d['contour_data'],
                levels=d.get('contour_levels'),
                colors=d.get('contour_colors', 'white'),
                linewidths=1,
                transform=self.data_crs,
                zorder=2
            )

        #  Inline labels — controlled by 'contour_labels' key
            if d.get('contour_labels', False):
                ax.clabel(
                    cs,
                    levels=d.get('contour_levels'),
                    inline=True,
                    inline_spacing=d.get('contour_label_spacing', 5),
                    fontsize=d.get('contour_label_fontsize', 7),
                    fmt=d.get('contour_label_fmt', '%g'),
                    colors=d.get('contour_colors', 'white'),
                    zorder=3
                )

    def _setup_ax(self, ax, background):

        ax.set_extent([self.x_min, self.x_max, self.y_min, self.y_max], crs=self.data_crs)
        ax.add_feature(cfeature.COASTLINE, zorder=3, edgecolor='black', linewidth=0.6)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1, color='grey', alpha=0.8, linestyle='--', zorder=5
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'white'}
        gl.ylabel_style = {'size': 10, 'color': 'white'}
        gl.xlocator   = plt.FixedLocator(range(-180, 181, 30))
        gl.ylocator   = plt.FixedLocator(range(-90, -59, 5))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # shared background
        ax.imshow(
            background['data'],
            origin='lower',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            transform=self.data_crs,
            cmap=background.get('cmap', 'gist_earth'), alpha=0.6, zorder=0
        )

        return gl
        
    def _add_scalebar(self, ax, length_km=200, n_segments=4, location=(0.08, 0.06), bar_height_frac=0.008):
        """
        Add a classic alternating black/white segmented scale bar.

        Parameters
        ----------
        ax             : cartopy GeoAxes
        length_km      : float, total length of the scale bar in kilometres
        n_segments     : int, number of alternating segments (even number recommended)
        location       : (x, y) axes-fraction coords for the LEFT end of the bar
        bar_height_frac: float, bar height as a fraction of the y-span
        """
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as pe

        x_span    = self.x_max - self.x_min
        y_span    = self.y_max - self.y_min
        total_m   = length_km * 1000
        seg_m     = total_m / n_segments
        seg_km    = length_km / n_segments
        bar_h     = y_span * bar_height_frac

        # Anchor: lower-left of the bar in projected coords
        x0 = self.x_min + location[0] * x_span
        y0 = self.y_min + location[1] * y_span

        colors = ['black', 'white']

        for i in range(n_segments):
            seg_x0 = x0 + i * seg_m
            seg_x1 = seg_x0 + seg_m
            color  = colors[i % 2]

            # Filled rectangle for each segment
            rect = mpatches.FancyBboxPatch(
                (seg_x0, y0), seg_m, bar_h,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor='black',
                linewidth=0.8,
                transform=self.data_crs,
                zorder=10
            )
            ax.add_patch(rect)

        # --- Outer border around the entire bar ---
        outer = mpatches.FancyBboxPatch(
            (x0, y0), total_m, bar_h,
            boxstyle="square,pad=0",
            facecolor='none',
            edgecolor='black',
            linewidth=1.2,
            transform=self.data_crs,
            zorder=11
        )
        ax.add_patch(outer)

        # --- Tick labels at each segment boundary ---
        outline = [pe.withStroke(linewidth=2, foreground='white')]

        for i in range(n_segments + 1):
            tick_x   = x0 + i * seg_m
            tick_km  = int(i * seg_km)

            # Small tick line above the bar
            ax.plot(
                [tick_x, tick_x],
                [y0 + bar_h, y0 + bar_h * 1.6],
                color='black', linewidth=0.8,
                transform=self.data_crs, zorder=12
            )

            # Label
            ax.text(
                tick_x, y0 + bar_h * 2.2,
                str(tick_km),
                ha='center', va='bottom',
                fontsize=7.5, color='black', fontweight='normal',
                transform=self.data_crs, zorder=13,
                path_effects=outline
            )

        # --- "km" unit label at the far right ---
        ax.text(
            x0 + total_m + x_span * 0.005, y0 + bar_h * 0.5,
            'km',
            ha='left', va='center',
            fontsize=8, color='black', fontweight='bold',
            transform=self.data_crs, zorder=13,
            path_effects=outline
        )

    def _add_background_colorbar(self, fig, ax, background):
        """
        Add a compact vertical colorbar for the background (hillshade) in the
        lower-right corner of the given axes.
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        vmin = background.get('vmin')
        vmax = background.get('vmax')
        if vmin is None:
            vmin = np.nanmin(background['data'])
        if vmax is None:
            vmax = np.nanmax(background['data'])

        norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap     = cm.get_cmap(background.get('cmap', 'gist_earth'))
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

        # Tall and thin — sits in the lower-right corner
        cax = inset_axes(
            ax,
            width='4%',        # thin
            height='40%',      # half the panel height
            loc='lower right',
            bbox_to_anchor=(-0.02, 0.04, 1.0, 1.0),  # nudge left so it clears the border
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        cb = fig.colorbar(mappable, cax=cax, orientation='vertical')

        cb.ax.yaxis.set_label_position('left')
        cb.ax.yaxis.tick_left()
        cb.ax.set_ylabel(
            background.get('cb_label', 'yes'),
            fontsize=10,
            labelpad=4,
            rotation=90,          # upright text running bottom-to-top
            va='center'
        )
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)

        for spine in cb.ax.spines.values():
            spine.set_linewidth(0.6)


        return cb

