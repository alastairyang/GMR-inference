import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio
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

    def plot(self, data, background):
        """ 
        Create (multipanel) plot
        
        data: tuple of dict
            data to be plotted. Tuple to accommodate multiple subplots
        background: dict
            background data (e.g., hillshade) to be plotted in all subplots
            
        both data and background are composed of dict with keys 
            'x', 'y', 'data', 
            'vmin','vmax', 
            'contour_data','contour_levels','contour_colors',
            'title','cb_label'
            if one of the option is not needed, it should be None


        As of June 7, 2026, we only plot 1 row 
        """

        n_panels = len(data)


        fig, axes = plt.subplots(1, n_panels,
            figsize=(5 * n_panels, 6),                      
            subplot_kw={'projection': self.proj}
        )

        # if one figure, make axes iterable
        if n_panels == 1:
            axes = [axes]

        for ax, d in zip(axes, data):
            # first ensure input data are defined on the same coordinate system. 
            d = self.make_consistent(d)

            self._setup_ax(ax, background)
            im0 = ax.imshow(
                d['data'],
                origin='lower',
                extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                transform=self.data_crs,
                cmap=d.get('cmap', 'viridis'),                         
                alpha=1, zorder=1,
                vmin=d.get('vmin'), vmax=d.get('vmax')
            )
            ax.set_title(d.get('title'))
            ax.set_aspect('equal')
            # add colorbar
            fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04,
                        label=d.get('cb_label', ''))
            # add dT contour
            if d.get('contour_data') is not None:
                x, y = np.meshgrid(d['x'], d['y'])
                cs1 = ax.contour(
                    x, y, d['contour_data'],
                    levels=d.get('contour_levels'),
                    colors=d.get('contour_colors', 'white'),
                    linewidths=1,
                    transform=self.data_crs,
                    zorder=2
                )
        # tight layout
        plt.tight_layout()
        
        return 
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

        # Hillshade (shared background)
        ax.imshow(
            background['data'],
            origin='upper',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            transform=self.data_crs,
            cmap=background.get('cmap', 'gist_earth'), alpha=0.6, zorder=0
        )

        return gl
