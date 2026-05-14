import numpy as np
import h5py
import geopandas as gpd
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import sklearn
import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures
import sklearn.preprocessing

data_file = "data.h5"

def plot(markers=None, predictions=None, title='', filename=None, vmin=None, vmax=None):
    """
    Plots weather prediction data.

    Args:
        markers (tuple or None): Tuple of (array of longitudinal coordinates, array of latidudinal coordinates, array of values) to plot. Shown as markers. If set to None, no markers will be shown.

        predictions (tuple or None): Tuple of (array of longitudinal coordinates, array of latidudinal coordinates, array of values) to plot. Shown as markers. If set to None, no markers will be shown. 

        title (str): Title for the plot

        filename (str or None): If set, the created plot will be saved to filename, otherwise shown.

        vmin (float): Lowest value for colormap, default=None
        vmax (float): Maximum value for colormap, default=None

    Returns:
        
    """
    ############################################################################
    ### (1) read in data on Austria's border                                 ###
    ############################################################################
    border = gpd.read_file('austria-detailed-boundary_854.geojson')
    border.set_crs(epsg=4326, inplace=True)

    ############################################################################
    ### (2) Data for plotting the markers                                    ###
    ############################################################################
    if markers is not None:
        df_markers = pd.DataFrame({
            'longitude': markers[0],
            'latitude': markers[1],
            'temperature': markers[2]
        })

        # Convert to GeoDataFrame
        gdf_markers = gpd.GeoDataFrame(df_markers, geometry=gpd.points_from_xy(df_markers.longitude, df_markers.latitude))
        gdf_markers.set_crs(epsg=4326, inplace=True)

    ############################################################################
    ### (2) Data for plotting the predictions                                ###
    ############################################################################
    if predictions is not None:
        df_predictions = pd.DataFrame({
            'longitude': predictions[0].ravel(),
            'latitude': predictions[1].ravel(),
            'temperature': predictions[2].ravel()
        })

        # Convert to GeoDataFrame
        gdf_predictions = gpd.GeoDataFrame(df_predictions, geometry=gpd.points_from_xy(df_predictions.longitude, df_predictions.latitude))
        gdf_predictions.set_crs(epsg=4326, inplace=True)

    ############################################################################
    ### Plot                                                                 ###
    ############################################################################
    fig, base = plt.subplots(1, 1)
    base.set_title(title)

    if predictions is not None:
        gdf_predictions_clipped = gdf_predictions.clip(mask=border)
        gdf_predictions_clipped.plot(ax=base, column='temperature', cmap='coolwarm', markersize=10, vmin=vmin, vmax=vmax)

    # show border
    border.boundary.plot(ax=base, edgecolor='black', color=None)

    if markers is not None:
        img = gdf_markers.plot(ax=base, column='temperature', cmap='coolwarm', markersize=10, vmin=vmin, vmax=vmax)

    if (markers is not None) or (predictions is not None):
        cmap = plt.cm.coolwarm

        # Add a colorbar with a custom location
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])

        fig.colorbar(sm, ax=base, location='right', pad=0.05, shrink=0.5)

        # Adjust layout to make space for colorbar
        plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')

class myPredictor:
    def __init__(self, degree=9):
        self.degree = degree
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.feature_map = PolynomialFeatures(degree=self.degree)
        self.predictor = sklearn.linear_model.Ridge()      

    def fit(self, X, y):
        Xt = self.scaler.fit_transform(X)
        Xt = self.feature_map.fit_transform(Xt)
        self.predictor.fit(Xt, y)

    def predict(self, X):
        Xt = self.scaler.transform(X)
        Xt = self.feature_map.transform(Xt)
        return self.predictor.predict(Xt)

if __name__ == "__main__":
    ## load data
    with h5py.File(data_file, 'r') as f:
        # <KeysViewHDF5 ['station_altitude', 'station_lat', 'station_lon', 'station_names', 'test', 'test_timesteps', 'train', 'train_timesteps']>
        station_altitudes =  f['station_altitude'][...]
        station_lat =  f['station_lat'][...]
        station_lon =  f['station_lon'][...]
        station_names =  f['station_names'][...]

        train_timesteps = f['train_timesteps'][...]
        train = f['train'][...]

        test_timesteps = f['test_timesteps'][...]
        test = f['test'][...]

        Lon, Lat = f['grid'][...]

    ## make predictions
    pred = myPredictor()
    pred.fit(np.hstack((station_lat.reshape(-1,1), station_lon.reshape(-1,1))), train[-1,:])
    Z = pred.predict(np.hstack((Lat.reshape(-1,1), Lon.reshape(-1,1))))
    Z = Z.reshape(Lat.shape)

    ## plot
    plot(
        title='My predictions',
        markers=(station_lon, station_lat, train[-1,:]),
        predictions=(Lon,Lat,Z),
        filename="plot.pdf",
        vmin=-10, vmax=10
    )
