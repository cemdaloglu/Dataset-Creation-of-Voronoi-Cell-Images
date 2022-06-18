import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
import os
import torch
import cv2
import torchvision.transforms as transforms
from geovoronoi import voronoi_regions_from_coords
from shapely.geometry import Polygon, mapping
from shapely.ops import cascaded_union, unary_union
import geopandas as gpd
import warnings
from typing import Union
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area, plot_line, \
    plot_polygon_collection_with_color, plot_voronoi_polys, plot_points

warnings.filterwarnings("ignore")


def vor_reg_creator(img_height: int, img_width: int, coords: list) -> Union[dict, Polygon]:
    '''
    Takes the desired size of a rectangle and random point coordinates to find the coordinates of the corner points
    of the Voronoi scheme. Returns the coordinates of those corner points and also the corner coordinates of the
    rectangle.

    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param coords: Coordinates of the random points, down left corner of the rectangle is the origin (0, 0).
    :return region_polys (dict[Polygon]): Dictionary of polygon objects size of the number of random points. Each
        element of the dictionary contains the coordinates of the corner points.
    :return boundary_shape (Polygon): A polygon object which contains the coordinates of the rectangle's corner points.
    '''

    area_max_lon = img_width
    area_min_lon = 0
    area_max_lat = img_height
    area_min_lat = 0

    lat_point_list = [area_min_lat, area_max_lat, area_max_lat, area_min_lat]
    lon_point_list = [area_min_lon, area_min_lon, area_max_lon, area_max_lon]

    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    boundary = gpd.GeoDataFrame()
    boundary = boundary.append({'geometry': polygon_geom}, ignore_index=True)

    boundary.crs = {'init': 'epsg:3395'}
    boundary_shape = unary_union(boundary.geometry)

    region_polys, region_pts = voronoi_regions_from_coords(np.array(tuple(coords)), boundary_shape)
    return region_polys, boundary_shape


def vor_plotting(coords: list, rand_point_no: int, region_polys: dict, boundary_shape: Polygon, img_width: int,
                 img_height: int, line_width: int, line_color: str, img_no: int) -> None:
    '''
    This function saves a black and white .png image with the desired line width, color, and name.

    :param coords: Coordinates of the random points, down left corner of the rectangle is the origin (0, 0).
    :param rand_point_no: Number of random points.
    :param region_polys: Dictionary of polygon objects size of the number of random points. Each element of the
        dictionary contains the coordinates of the corner points.
    :param boundary_shape: A polygon object which contains the coordinates of the rectangle's corner points.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param line_width: Width of the line, 0 means no line.
    :param line_color: Color of the line.
    :param img_no: The number that the current image will be saved with a name with that number.
    '''

    fig, ax = subplot_for_map(figsize=(img_width / 120, img_height / 120))
    crds = np.array(tuple(coords))
    for point_no in range(rand_point_no):
        points = []
        for p_idx in range(len(region_polys[point_no].exterior.coords.xy[0])):
            points.append([region_polys[point_no].exterior.coords.xy[0][p_idx],
                           region_polys[point_no].exterior.coords.xy[1][p_idx]])
        points = np.array(tuple(points))
        plot_line(ax, points, linewidth=line_width, color=line_color)
    plot_voronoi_polys_with_points_in_area(ax, boundary_shape, region_polys, crds, points_color='white',
                                           voronoi_color='white', voronoi_edgecolor='blue')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('Image {}'.format(img_no + 1) + '.png', dpi=120, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()


def vor_plotting_rgb(rand_point_no: int, region_polys: dict, img_width: int,
                     img_height: int, line_width: int, line_color: str, img_no: int) -> None:
    '''
    This function saves a colored .png image with the desired line width, color, and name.

    :param rand_point_no: Number of random points.
    :param region_polys: Dictionary of polygon objects size of the number of random points. Each element of the
      dictionary contains the coordinates of the corner points.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param line_width: Width of the line, 0 means no line.
    :param line_color: Color of the line.
    :param img_no: The number that the current image will be saved with a name with that number.
    '''

    fig, ax = subplot_for_map(figsize=(img_width / 120, img_height / 120))
    for point_no in range(rand_point_no):
        points = []
        for p_idx in range(len(region_polys[point_no].exterior.coords.xy[0])):
            points.append([region_polys[point_no].exterior.coords.xy[0][p_idx],
                           region_polys[point_no].exterior.coords.xy[1][p_idx]])
        points = np.array(tuple(points))
        plot_line(ax, points, linewidth=line_width, color=line_color)
    rand_color = random.sample(list(matplotlib.colors.CSS4_COLORS), k=rand_point_no)
    for point_no in range(rand_point_no):
        plot_polygon_collection_with_color(ax, [region_polys[point_no]], color=rand_color[point_no])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('Image_rgb {}'.format(img_no + 1) + '.png', dpi=120, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()


def vor_bw_img_creator(total_img: int, rand_point_no: int, img_height: int, img_width: int, line_width: int,
                       line_color: str) -> None:
    '''
    Takes how many black and white Voronoi images that will be constructed and creates the desired amount of random
    points for each Voronoi image. Calls the vor_reg_creator and vor_plotting functions.

    :param total_img: Number of Voronoi images that will be constructed.
    :param rand_point_no: Number of random points.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param line_width: Width of the line, 0 means no line.
    :param line_color: Color of the line.
    '''

    for img_no in range(total_img):
        coords = []
        for point_no in range(rand_point_no):
            y_axis = random.uniform(0.0, img_height)
            x_axis = random.uniform(0.0, img_width)
            coords.append([x_axis, y_axis])
        region_polys, boundary_shape = vor_reg_creator(img_height, img_width, coords)
        vor_plotting(coords, rand_point_no, region_polys, boundary_shape, img_width, img_height, line_width, line_color,
                     img_no)


"""def vor_rgb_img_creator(total_img: int, rand_point_no: int, img_height: int, img_width: int, line_width: int,
                        line_color: str) -> None:
    '''
    Takes how many colored Voronoi images that will be constructed and creates the desired amount of random points for
    each Voronoi image. Calls the vor_reg_creator and vor_plotting functions.

    :param total_img: Number of Voronoi images that will be constructed.
    :param rand_point_no: Number of random points.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param line_width: Width of the line, 0 means no line.
    :param line_color: Color of the line.
    '''

    for img_no in range(total_img):
        coords = []
        for point_no in range(rand_point_no):
            y_axis = random.uniform(0.0, img_height)
            x_axis = random.uniform(0.0, img_width)
            coords.append([x_axis, y_axis])
        region_polys, boundary_shape = vor_reg_creator(img_height, img_width, coords)
        vor_plotting_rgb(rand_point_no, region_polys, img_width, img_height, line_width,
                         line_color, img_no)"""


def img_to_tensor_bw(img: str, img_height: int, img_width: int) -> torch.Tensor:
    '''
    Takes the filename of a black and white image and returns a tensor.

    :param img: Filename of the image whose tensors will be returned.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :return tensor (torch.Tensor): Tensor of the given image, return shape is (img_width, img_height)
    '''

    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(img_width, img_height, 1)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image)
    return tensor


def img_to_tensor_colored(img: str, img_height: int, img_width: int) -> torch.Tensor:
    '''
    Takes the filename of a colored image and returns a tensor.

    :param img: Filename of the image whose tensors will be returned.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :return tensor: Tensor of the given image, return shape is (3, img_width, img_height)
    '''

    image = cv2.imread(img)
    image = image.reshape(img_width, img_height, 3)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image)
    return tensor


def tensor_out_bw(total_img: int, img_height: int, img_width: int) -> torch.Tensor:
    '''
    Takes the total number of bw images and returns the tensor of each image.

    :param total_img: Filename of the image whose tensors will be returned.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :return tensor: Tensor of the all images, return shape is (total_img, 1, img_width, img_height)
    '''

    output_bw = img_to_tensor_bw('Image {}'.format(1) + '.png', img_height, img_width)
    for idx in range(total_img - 1):
        output_bw = torch.cat((output_bw, img_to_tensor_bw('Image {}'.format(idx + 2) + '.png', img_height, img_width)))
    output_bw = output_bw.reshape(total_img, 1, img_width, img_height)
    return output_bw


def tensor_out_colored(total_img: int, img_height: int, img_width: int) -> torch.Tensor:
    '''
    Takes the total number of colored images and returns the tensor of each image.

    :param total_img: Filename of the image whose tensors will be returned.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :return tensor: Tensor of the all images, return shape is (total_img, 3, img_width, img_height)
    '''

    output_rgb = img_to_tensor_colored('Image_rgb {}'.format(1) + '.png', img_height, img_width)
    for idx in range(total_img - 1):
        output_rgb = torch.cat(
            (output_rgb, img_to_tensor_colored('Image_rgb {}'.format(idx + 2) + '.png', img_height, img_width)))
    output_rgb = output_rgb.reshape(total_img, 3, img_width, img_height)
    return output_rgb


def tensor_out_2(total_img: int, img_height: int, img_width: int, line_width: int) -> torch.Tensor:
    '''
    Takes the total number of colored images and returns a tensor which contains unique values for each Voronoi
    pixel, and 0 for the border pixels.

    :param total_img: Filename of the image whose tensors will be returned.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param line_width: Width of the line.
    :return tensor (): Tensor of the all images, return shape is (total_img, img_width, img_height)
    '''
    img_gray_mode = cv2.imread('Image_rgb {}'.format(1) + '.png', 0)
    img_gray_mode[:line_width, :] = 0
    img_gray_mode[:, :line_width] = 0
    img_gray_mode[img_gray_mode.shape[0] - line_width:, :] = 0
    img_gray_mode[:, img_gray_mode.shape[1] - line_width:] = 0
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img_gray_mode)

    for idx in range(total_img - 1):
        img_gray_mode2 = cv2.imread('Image_rgb {}'.format(idx + 2) + '.png', 0)
        img_gray_mode2[:line_width, :] = 0
        img_gray_mode2[:, :line_width] = 0
        img_gray_mode2[img_gray_mode.shape[0] - line_width:, :] = 0
        img_gray_mode2[:, img_gray_mode.shape[1] - line_width:] = 0
        tensor2 = transform(img_gray_mode2)
        tensor = torch.cat((tensor, tensor2))
    tensor = tensor.reshape(total_img, img_width, img_height)
    return tensor


def vor_rgb_img_creator(total_img: int, rand_point_no: int, img_height: int, img_width: int, line_width: int,
                        line_color: str) -> torch.Tensor:
    '''
    Takes how many colored Voronoi images that will be constructed and creates the desired amount of random points for each Voronoi
    image. Calls the vor_reg_creator and vor_plotting functions.

    :param total_img (int): Number of Voronoi images that will be constructed.
    :param rand_point_no (int): Number of random points.
    :param img_height (int): Height of the image.
    :param img_widht (int): Width of the image.
    :param line_width (int): Width of the line, 0 means no line.
    :param line_color (str): Color of the line.
    '''
    for img_no in range(total_img):
        coords = []
        for point_no in range(rand_point_no):
            y_axis = random.uniform(0.0, img_height)
            x_axis = random.uniform(0.0, img_width)
            coords.append([x_axis, y_axis])
        region_polys, boundary_shape = vor_reg_creator(img_height, img_width, coords)
        vor_plotting_rgb(rand_point_no, region_polys, img_width, img_height, line_width,
                         line_color, img_no)
        vor_plotting_rgb_mock(rand_point_no, region_polys, img_width, img_height, line_width,
                              line_color, img_no)
    tensor3_1 = tensor_out_colored(total_img, img_height, img_width)
    tensor3_2 = tensor_out_colored_mock(total_img, img_height, img_width)
    tensor3 = (tensor3_1 + tensor3_2) / 2

    for idx in range(total_img):
        os.remove('Image_rgb_mock {}'.format(idx + 1) + '.png')
    return tensor3


def vor_plotting_rgb_mock(rand_point_no: int, region_polys: dict, img_width: int,
                     img_height: int, line_width: int, line_color: str, img_no: int) -> None:
    '''
    This function saves a colored .png image with the desired line width, color, and name.

    :param rand_point_no: Number of random points.
    :param region_polys: Dictionary of polygon objects size of the number of random points. Each element of the
      dictionary contains the coordinates of the corner points.
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param line_width: Width of the line, 0 means no line.
    :param line_color: Color of the line.
    :param img_no: The number that the current image will be saved with a name with that number.
    '''

    fig, ax = subplot_for_map(figsize=(img_width / 120, img_height / 120))
    for point_no in range(rand_point_no):
        points = []
        for p_idx in range(len(region_polys[point_no].exterior.coords.xy[0])):
            points.append([region_polys[point_no].exterior.coords.xy[0][p_idx],
                           region_polys[point_no].exterior.coords.xy[1][p_idx]])
        points = np.array(tuple(points))
        plot_line(ax, points, linewidth=line_width, color=line_color)
    rand_color = random.sample(list(matplotlib.colors.CSS4_COLORS), k=rand_point_no)
    for point_no in range(rand_point_no):
        plot_polygon_collection_with_color(ax, [region_polys[point_no]], color=rand_color[point_no])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('Image_rgb_mock {}'.format(img_no + 1) + '.png', dpi=120, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()


def tensor_out_colored_mock(total_img: int, img_height: int, img_width: int) -> torch.Tensor:
    '''
    Takes the total number of colored images and returns the tensor of each image.

    :param total_img (int): Filename of the image whose tensors will be returned.
    :param img_height (int): Height of the image.
    :param img_widht (int): Width of the image.
    :return tensor (): Tensor of the all images, return shape is (total_img, 3, img_width, img_height)
    '''
    output = img_to_tensor_colored('Image_rgb_mock {}'.format(1)+'.png', img_height, img_width)
    for idx in range(total_img-1):
        output = torch.cat((output, img_to_tensor_colored('Image_rgb_mock {}'.format(idx+2)+'.png', img_height, img_width)))
    output = output.reshape(total_img, 3, img_width, img_height)
    return output