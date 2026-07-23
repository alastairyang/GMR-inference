import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import xy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely.geometry as sgeom
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
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.x_min = np.min(x) - dx / 2
        self.x_max = np.max(x) + dx / 2
        self.y_min = np.min(y) - dy / 2
        self.y_max = np.max(y) + dy / 2
        self.x = x
        self.y = y

    def make_consistent(self, data_dict):
        """
        Check coordinate. If not consistent, interpolate.
        Scatter panels are passed through unchanged.

        data_dict: dict with keys 'x', 'y', 'data'
        """
        if data_dict.get('data_type') == 'scatter':
            return data_dict  # point cloud — skip grid interpolation entirely

        # Also skip if using layers (mixed types handled per-layer)
        if 'layers' in data_dict:
            return data_dict

        x, y = data_dict['x'], data_dict['y']
        if not (np.array_equal(x, self.x) and np.array_equal(y, self.y)):
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
            self._add_lon_labels(ax, lons=[-120, -100, -140])
            if d.get('scalebar') is not None:
                self._add_scalebar(ax, length_km=d['scalebar'][0], n_segments=d['scalebar'][1])
            self._plot_panel(fig, ax, d)

        # Add background colorbar on the last (rightmost) panel, consistent with main+stack
        self._add_background_colorbar(fig, axes[-1], background)

        plt.tight_layout()
        return fig

    # private: main + two stacked layout
    def _plot_main_stack(self, data, background):
        assert len(data) == 3, "'main+stack' layout requires exactly 3 panels in data."

        fig = plt.figure(figsize=(13, 8))

        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[2, 1],
            height_ratios=[1, 1],
            wspace=0.08,
            hspace=0.15
        )

        ax_main   = fig.add_subplot(gs[0:2, 0], projection=self.proj)
        ax_top    = fig.add_subplot(gs[0,   1], projection=self.proj)
        ax_bottom = fig.add_subplot(gs[1,   1], projection=self.proj)

        for ax, d in zip([ax_main, ax_top, ax_bottom], data):
            d = self.make_consistent(d)
            self._setup_ax(ax, background)
            self._add_lon_labels(ax, lons=[-120, -100, -140])
            if d.get('scalebar') is not None:
                self._add_scalebar(ax, length_km=d['scalebar'][0], n_segments=d['scalebar'][1])
            self._plot_panel(fig, ax, d)

        self._add_background_colorbar(fig, ax_main, background)

        ax_main.set_label('main')
        ax_top.set_label('top')
        ax_bottom.set_label('bottom')

        return fig


    def _plot_panel(self, fig, ax, d):
        if 'layers' in d:
            for i, layer in enumerate(d['layers']):
                self._draw_layer(fig, ax, d, layer, zorder_offset=i)
        else:
            # Single-layer: treat the dict itself as the layer
            self._draw_layer(fig, ax, d, d, zorder_offset=0)

        ax.set_title(d.get('title', ''), fontsize=10)
        ax.set_aspect('equal')

        # Add legend if any layer has a label
        if d.get('legend', True):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles, labels,
                    loc=d.get('legend_loc', 'lower left'),
                    fontsize=8,
                    markerscale=2,
                    framealpha=0.7,
                    edgecolor='gray',
                )

    def _draw_layer(self, fig, ax, panel_d, layer_d, zorder_offset=0):
        """Dispatch a single layer to the correct draw method."""
        data_type = layer_d.get('data_type', 'pcolor')

        if data_type == 'scatter':
            self._draw_scatter_layer(fig, ax, panel_d, layer_d, zorder_offset)
        elif data_type == 'line':
            self._draw_line_layer(ax, layer_d, zorder_offset)
        elif data_type == 'pcolor':
            self._draw_pcolor_layer(fig, ax, panel_d, layer_d)
        else:
            raise ValueError(f"Unknown data_type: '{data_type}'")

    def _draw_scatter_layer(self, fig, ax, panel_d, layer_d, zorder_offset=0):
        """Draw a single scatter layer, extracted from _plot_panel_scatter."""
        SCATTER_KEYS = ('s', 'marker', 'alpha', 'linewidths', 'zorder', 'label')
        kw = {k: layer_d[k] for k in SCATTER_KEYS if k in layer_d}
        kw.setdefault('s', 10)
        kw.setdefault('marker', 'o')
        kw.setdefault('alpha', 0.8)
        kw.setdefault('linewidths', 0.4)
        kw.setdefault('zorder', 2 + zorder_offset)

        x_pts = np.asarray(layer_d['x'])
        y_pts = np.asarray(layer_d['y'])

        if layer_d.get('facecolors') == 'none':
            # Hollow circles — no colormap
            ax.scatter(
                x_pts, y_pts,
                facecolors='none',
                edgecolors=layer_d.get('edgecolors', 'black'),
                transform=self.data_crs,
                **kw
            )

        elif 'data' in layer_d:
            # Colour-mapped fill
            values = np.asarray(layer_d['data'])
            sc = ax.scatter(
                x_pts, y_pts,
                c=values,
                cmap=layer_d.get('cmap', 'viridis'),
                vmin=layer_d.get('vmin'),
                vmax=layer_d.get('vmax'),
                edgecolors=layer_d.get('edgecolors', 'none'),
                transform=self.data_crs,
                **kw
            )
            if panel_d.get('colorbar', 'on') != 'off':
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                             label=layer_d.get('cb_label', panel_d.get('cb_label', '')))

        else:
            # Solid fill with facecolor
            ax.scatter(
                x_pts, y_pts,
                facecolors=layer_d.get('facecolors', 'blue'),
                edgecolors=layer_d.get('edgecolors', 'none'),
                transform=self.data_crs,
                **kw
            )

    def _draw_line_layer(self, ax, layer_d, zorder_offset=0):
        LINE_KEYS = ('linewidth', 'linestyle', 'alpha', 'color', 'label')
        kw = {k: layer_d[k] for k in LINE_KEYS if k in layer_d}
        kw.setdefault('linewidth', 1.0)
        kw.setdefault('linestyle', '-')
        kw.setdefault('alpha',     1.0)
        kw.setdefault('color',     'black')
        kw.setdefault('zorder',    3 + zorder_offset)

        ax.plot(np.asarray(layer_d['x']),
                np.asarray(layer_d['y']),
                transform=self.data_crs, **kw)

    def _draw_pcolor_layer(self, fig, ax, panel_d, layer_d):
        """Draw a single pcolor/imshow layer."""
        im = ax.imshow(
            layer_d['data'],
            origin='lower',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            transform=self.data_crs,
            cmap=layer_d.get('cmap', 'viridis'),
            alpha=layer_d.get('alpha', 1.0),
            zorder=layer_d.get('zorder', 1),
            vmin=layer_d.get('vmin'),
            vmax=layer_d.get('vmax')
        )

        if panel_d.get('colorbar', 'on') != 'off':
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=layer_d.get('cb_label', panel_d.get('cb_label', '')))

        if layer_d.get('contour_data') is not None:
            x, y = np.meshgrid(layer_d['x'], layer_d['y'])
            cs = ax.contour(
                x, y, layer_d['contour_data'],
                levels=layer_d.get('contour_levels'),
                colors=layer_d.get('contour_colors', 'white'),
                linewidths=1,
                transform=self.data_crs,
                zorder=2
            )
            if layer_d.get('contour_labels', False):
                ax.clabel(
                    cs,
                    levels=layer_d.get('contour_levels'),
                    inline=True,
                    inline_spacing=layer_d.get('contour_label_spacing', 5),
                    fontsize=layer_d.get('contour_label_fontsize', 7),
                    fmt=layer_d.get('contour_label_fmt', '%g'),
                    colors=layer_d.get('contour_colors', 'white'),
                    zorder=3
                )

    def _plot_panel_image(self, fig, ax, d):
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

    def _plot_panel_line(self, fig, ax, d):
        LINE_KEYS = ('linewidth', 'linestyle', 'alpha', 'zorder', 'color')

        if 'layers' in d:
            for layer in d['layers']:
                self._draw_line_layer(ax, layer)
        else:
            self._draw_line_layer(ax, d)

        ax.set_title(d.get('title', ''), fontsize=10)
        ax.set_aspect('equal')

    def _plot_panel_scatter(self, fig, ax, d):
        SCATTER_KEYS = ('s', 'marker', 'alpha', 'linewidths', 'zorder')

        def _draw_layer(layer_d, zorder_offset=0):
            kw = {k: layer_d[k] for k in SCATTER_KEYS if k in layer_d}
            kw.setdefault('s', 10)
            kw.setdefault('marker', 'o')
            kw.setdefault('alpha', 0.8)
            kw.setdefault('linewidths', 0.4)
            kw.setdefault('zorder', 2 + zorder_offset)

            x_pts = np.asarray(layer_d['x'])
            y_pts = np.asarray(layer_d['y'])

            if layer_d.get('facecolors') == 'none':
                ax.scatter(
                    x_pts, y_pts,
                    facecolors='none',
                    edgecolors=layer_d.get('edgecolors', 'black'),
                    transform=self.data_crs,
                    **kw
                )
                return None  

            elif 'data' in layer_d:
                values = np.asarray(layer_d['data'])
                sc = ax.scatter(
                    x_pts, y_pts,
                    c=values,
                    cmap=layer_d.get('cmap', 'viridis'),
                    vmin=layer_d.get('vmin'),
                    vmax=layer_d.get('vmax'),
                    edgecolors=layer_d.get('edgecolors', 'none'),
                    transform=self.data_crs,
                    **kw
                )
                return sc
            else:
                ax.scatter(
                    x_pts, y_pts,
                    facecolors=layer_d.get('facecolors', 'blue'),
                    edgecolors=layer_d.get('edgecolors', 'none'),
                    transform=self.data_crs,
                    **kw
                )
                return None

        if 'layers' in d:
            for i, layer in enumerate(d['layers']):
                sc = _draw_layer(layer, zorder_offset=i)
                if sc is not None and d.get('colorbar', 'on') != 'off':
                    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                                label=layer.get('cb_label', d.get('cb_label', '')))
        else:
            sc = _draw_layer(d)
            if sc is not None and d.get('colorbar', 'on') != 'off':
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                            label=d.get('cb_label', ''))

        ax.set_title(d.get('title'), fontsize=10)
        ax.set_aspect('equal')

    def _setup_ax(self, ax, background):

        ax.set_extent([self.x_min, self.x_max, self.y_min, self.y_max], crs=self.data_crs)
        ax.add_feature(cfeature.COASTLINE, zorder=3, edgecolor='black', linewidth=0.6)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1, color='black', alpha=0.8, linestyle='--', zorder=5
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}
        gl.xlocator   = plt.FixedLocator(range(-180, 181, 20))
        gl.ylocator   = plt.FixedLocator(range(-90, -59, 3))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        gl.xlabel_style = {
            'size': 10,
            'color': 'black',
            'rotation': 0,
            'ha': 'center',
            'va': 'top',
        }
        gl.bottom_labels = False
        gl.top_labels    = False
        gl.right_labels  = False
        gl.left_labels   = True

        ax.imshow(
            background['data'],
            origin='lower',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            transform=self.data_crs,
            cmap=background.get('cmap', 'gist_earth'), alpha=background.get('alpha', 0.6), zorder=0
        )

        return gl

    def _add_lon_labels(self, ax, lons):
        """Place longitude labels where gridlines exit through bottom or right boundary."""
        for lon in lons:
            pts = [self.data_crs.transform_point(lon, lat, ccrs.PlateCarree())
                for lat in np.linspace(-90, -60, 300)]

            pts = [(x, y) for x, y in pts
                if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max]

            if not pts:
                continue

            bottom_pts = [(x, y) for x, y in pts
                        if abs(y - self.y_min) < (self.y_max - self.y_min) * 0.15]
            right_pts  = [(x, y) for x, y in pts
                        if abs(x - self.x_max) < (self.x_max - self.x_min) * 0.15]

            if bottom_pts:
                lx, ly = min(bottom_pts, key=lambda p: p[1])
                ha, va = 'center', 'top'
            elif right_pts:
                lx, ly = max(right_pts, key=lambda p: p[0])
                ha, va = 'left', 'center'
            else:
                lx, ly = pts[-1]
                ha, va = 'center', 'top'

            label = f"{abs(lon)}°{'W' if lon < 0 else 'E'}"
            ax.text(lx, ly, label,
                    transform=self.data_crs,
                    fontsize=10, ha=ha, va=va, zorder=6)

    def _add_scalebar(self, ax, length_km=200, n_segments=4, location=(0.08, 0.06), bar_height_frac=0.008):
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as pe

        x_span    = self.x_max - self.x_min
        y_span    = self.y_max - self.y_min
        total_m   = length_km * 1000
        seg_m     = total_m / n_segments
        seg_km    = length_km / n_segments
        bar_h     = y_span * bar_height_frac

        x0 = self.x_min + location[0] * x_span
        y0 = self.y_min + location[1] * y_span

        colors = ['black', 'white']

        for i in range(n_segments):
            seg_x0 = x0 + i * seg_m
            seg_x1 = seg_x0 + seg_m
            color  = colors[i % 2]

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

        outline = [pe.withStroke(linewidth=2, foreground='white')]

        for i in range(n_segments + 1):
            tick_x   = x0 + i * seg_m
            tick_km  = int(i * seg_km)

            ax.plot(
                [tick_x, tick_x],
                [y0 + bar_h, y0 + bar_h * 1.6],
                color='black', linewidth=0.8,
                transform=self.data_crs, zorder=12
            )

            ax.text(
                tick_x, y0 + bar_h * 2.2,
                str(tick_km),
                ha='center', va='bottom',
                fontsize=7.5, color='black', fontweight='normal',
                transform=self.data_crs, zorder=13,
                path_effects=outline
            )

        ax.text(
            x0 + total_m + x_span * 0.005, y0 + bar_h * 0.5,
            'km',
            ha='left', va='center',
            fontsize=8, color='black', fontweight='bold',
            transform=self.data_crs, zorder=13,
            path_effects=outline
        )

    def _add_background_colorbar(self, fig, ax, background):
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

        cax = inset_axes(
            ax,
            width='4%',
            height='40%',
            loc='lower right',
            bbox_to_anchor=(-0.02, 0.04, 1.0, 1.0),
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
            rotation=90,
            va='center'
        )
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)

        for spine in cb.ax.spines.values():
            spine.set_linewidth(0.6)

        return cb
