import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2
import plotly.graph_objects as go
import imageio
import base64
import io
from PIL import Image


def interpolate_closed_curve(x, y, num_points=3000):
    """
    Interpolates a closed 2D curve through (x, y) data using a periodic spline.
    """
    tck, u = interpolate.splprep([x, y], s=0, per=True) # consider this as periodic data, to draw close curve
    # tck, u = interpolate.splprep([x, y], s=0, per=False)
    
    u_interp = np.linspace(0, 1, num_points)
    x_interp, y_interp = interpolate.splev(u_interp, tck)
    
    return x_interp, y_interp


def smoothen_trajectory(x, y, start_idx, end_idx, window_size=101, num_iter=20):
    """ Smoothen a interval of the trajectory by moving average

    Args:
        x: np.array shape (n, )
        y: np.array shape (n, )
        start_idx: start index of the interval
        end_idx: end index of the interval 
        window_size: must be odd
        num_iter: number of filtering interations, NOTE: increase this to receive more intersive smoothening 

    Return:
        np.array: trajectory with the specified segment smoothened
    """
    assert (window_size % 2) == 1, "window size must be odd"
    # if start_idx - window_size//2 < 0 or end_idx + window_size//2 > len(x):
    #     raise IndexError("Index range with window extension is out of bounds.")
        
    x_smooth = x.copy()
    y_smooth = y.copy()
    
    kernel = np.ones(window_size) / window_size
    for i in range(num_iter):
        if start_idx - window_size//2 < 0 or end_idx + window_size//2 > len(x):
            x_temp = np.concatenate([x_smooth[start_idx - window_size//2:-1], 
                                     x_smooth[0:end_idx + window_size//2]])
            y_temp = np.concatenate([y_smooth[start_idx - window_size//2:-1], 
                                     y_smooth[0:end_idx + window_size//2]])
            pass
        else:
            x_smooth[start_idx:end_idx] = np.convolve(x_smooth[start_idx - window_size//2:end_idx + window_size//2], 
                                                    kernel, mode='valid')  
            y_smooth[start_idx:end_idx] = np.convolve(y_smooth[start_idx - window_size//2:end_idx + window_size//2], 
                                                    kernel, mode='valid')
    return x_smooth, y_smooth


class MapIntergratedVisualizer:
    def __init__(self, img_file, origin, resolution):
        self.img_file = img_file
        self.origin = origin
        self.resolution = resolution

    def init_map(self):
        img = imageio.imread(self.img_file)
        img_height, img_width = img.shape[:2]

        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        encoded_img = base64.b64encode(buf.getvalue()).decode("ascii")
        data_uri = "data:image/png;base64," + encoded_img

        origin_x, origin_y = self.origin[0], self.origin[1]
        res = self.resolution            

        map_width_meters = img_width * res
        map_height_meters = img_height * res

        # create map
        fig = go.Figure()
        fig.update_layout(
            images=[dict(
                source=data_uri,
                x=origin_x, y=origin_y + map_height_meters, # coordinate of the top-left of image
                sizex=map_width_meters, sizey=map_height_meters,
                xref="x",
                yref="y",
                layer="below"
            )],

            xaxis=dict(
                visible=True,
                range=[origin_x, origin_x + map_width_meters],
                scaleanchor="y",  # same scale for x & y
                autorange=True
            ),
            yaxis=dict(
                visible=True,
                range=[origin_y, origin_y + map_height_meters],
                autorange=True
            ),

            width=600,
            height=800,

            margin=dict(l=0, r=0, t=30, b=0),
            dragmode='pan'
        )
        return fig

    def world2pixel_coord(self, x, y):
        x_pixels = [(xi - self.origin[0]) / self.resolution for xi in x]
        y_pixels = [(yi - self.origin[1]) / self.resolution for yi in y]
        return x_pixels, y_pixels

    def pixeld2word_coord(self, x_pixels, y_pixels):
        x = [xi * self.resolution + self.origin[0] for xi in x_pixels]
        y = [yi * self.resolution + self.origin[1] for yi in y_pixels]
        return x, y

    def add_scatter_points(self,fig, x, y, color='blue', size=3, name='Points'):
        hover_texts = [
            f"Idx: {i}<br>X: {xp:.2f}, Y: {yp:.2f}"
            for i, xp, yp in zip(range(len(x)), x, y)
        ]
        scatter = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color=color, size=size),
            name=name,
            hovertext=hover_texts,
            hoverinfo="text"
        )
        fig.add_trace(scatter)
        return fig

    def add_yaw_arrows(self, fig, x, y, yaw, arrow_length=0.75, color='red'):
        for xp, yp, theta in zip(x, y, yaw):
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            fig.add_annotation(
                x=xp + dx, y=yp + dy,
                ax=xp, ay=yp,
                xref="x", yref="y",
                axref="x",ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1, arrowwidth=1,
                arrowcolor=color,
                text=""  # no text
            )
        return fig

    def add_speed_distribution(self, fig, x, y, speed, cmap='viridis'):
        hover_texts = [
            f"Idx: {i} Speed {speed:.2f}<br>X: {xp:.2f}, Y: {yp:.2f}"
            for i, xp, yp, speed in zip(range(len(x)), x, y, speed)
        ]
        
        scatter_trace = go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(color=speed, colorscale=cmap, size=3, colorbar=dict(title='Speed (m/s)')),
            hovertext=hover_texts,
            hoverinfo="text"
        )
        fig.add_trace(scatter_trace)
        fig.update_layout(title='Speed distribution')
        return fig


class PointsExtractor:
    def __init__(self, img_file, color):
        """
        Args:
            color (tuple-like): (R,G,B), specified the color of the points extracted
        """
        self.img_file = img_file
        self.color = color

    def extract_points_unsorted(self):
        """
        Extract unsorted pixel coordinates matching the target color.
        """
        img = cv2.imread(self.img_file)  # Update the path as needed
        print(img.shape)
        img_array = np.array(img) # (B, G, R)
        img_array = img_array[::-1, :, ::-1] # (R, G, B) and y-axis bottom-up

        R, G, B = self.color[0], self.color[1], self.color[2]
        mask = (img_array[:,:,0] == R) & (img_array[:,:,1] == G) & (img_array[:,:,2] == B)
        unsorted_points = np.argwhere(mask)
        unsorted_points = unsorted_points[:, ::-1]
        return unsorted_points
    
    def sort_points_closed_curve(self, points):
        """
        Sort a set of points that form a closed curve in counterclockwise order
        """
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
        sorted_indices = np.argsort(angles)
        points_sorted = points[sorted_indices]
        return points_sorted
    
    def extract_points(self):
        """
        Extract pixel coordinates matching the target color, counterclockwise order
        """
        unsorted_points = self.extract_points_unsorted()
        sorted_points = self.sort_points_closed_curve(unsorted_points)
        return sorted_points
