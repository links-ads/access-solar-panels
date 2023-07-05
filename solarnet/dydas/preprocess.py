import numpy as np
import pyproj as prj
from rasterio import features
from rasterio import windows as win
from rasterio import DatasetReader, MemoryFile
from shapely.geometry import Polygon, MultiPolygon, ops
from typing import Any, List, Union


def project(shape: Polygon, from_crs: prj.CRS, to_crs: prj.CRS = prj.CRS(3857)) -> Polygon:
    """Projects the given vector shape from the source CRS to the destination CRS, using
    pyproject and Shapely.

    Args:
        shape (Polygon): input Shapely object to be projected
        from_crs (prj.CRS): source Coord. Reference System of the input shape
        to_crs (prj.CRS, optional): destination CRS. Defaults to EPSG:3857 (Mercator).

    Returns:
        Polygon: same shape, but reprojected.
    """
    projection = prj.Transformer.from_crs(from_crs, to_crs, always_xy=True).transform
    return ops.transform(projection, shape)


def box_dimentions(shape: Polygon, crs: prj.CRS, resolution: float = 1.0):
    """Estimates the size, in pixels, of the given shape, provided the corresponding CRS and resolution.
    WARNING: The result may slightly differ from other tools such as rasterio, use this at your own risk.

    Args:
        shape (Polygon): input polygon, or Shapely object in general
        crs (prj.CRS): CRS of the input shape
        resolution (float, optional): resolution in metres per pixel, considering each side. Defaults to 1.0.

    Returns:
        Tuple[int, int]: estimated pixel height and width of the rasterized shape for that resolution
    """
    if crs != prj.CRS(3857):
        mercator_shape = project(shape, from_crs=crs)
    else:
        mercator_shape = shape
    minx, miny, maxx, maxy = mercator_shape.bounds
    resx, resy = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    return np.ceil(np.abs(maxx - minx) / resx), np.ceil(np.abs(maxy - miny) / resy)


def tile_overlapped(image: np.ndarray, tile_size: Union[tuple, int] = 256, channels_first: bool = False) -> np.ndarray:
    """Divides the input image into tiles with fixed size, computing the overlap from the remainder.
    The input must be a single- or multi-channel image, the output is always a tensor with size
    [tilesX, tilesY, tile height, tile width, channels]

    Args:
        image (np.ndarray): input image [H, W] or [H, W, C] or [C, H, W]
        tile_size (Union[tuple, int], optional): Tile size, a single value if squared or a tuple for height and width. Defaults to 256.
        channels_first (bool, optional): Whether it is channels-first or not. Defaults to False.

    Raises:
        ValueError: When the image is smaller than the tile size

    Returns:
        np.ndarray: numpy array with dimensions [nTH, nTW, TH, TW, C]
    """
    if channels_first:
        image = np.moveaxis(image, 0, -1)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    # assume height, width, channels from now on
    height, width, channels = image.shape
    tile_h, tile_w = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    if height <= tile_h and width <= tile_w:
        raise ValueError("Image is smaller than the required tile size")
    # number of expected tiles
    tile_count_h = int(np.ceil(width / tile_h))
    tile_count_w = int(np.ceil(height / tile_w))
    # compute total remainder for the expanded window
    remainder_h = (tile_count_h * tile_h) - height
    remainder_w = (tile_count_w * tile_w) - width
    # divide remainders among tiles as overlap (floor to keep overlap to the minimum)
    overlap_h = int(np.floor(remainder_h / float(tile_count_h - 1))) if tile_count_h > 1 else 0
    overlap_w = int(np.floor(remainder_w / float(tile_count_w - 1))) if tile_count_w > 1 else 0
    # create the empty tensor to contain tiles
    tiles = np.empty((tile_count_h, tile_count_w, tile_h, tile_w, channels), dtype=image.dtype)
    for row in range(tile_count_h):
        for col in range(tile_count_w):
            # get the starting indices, accounting from initial positions
            x = max(row * tile_h - overlap_h, 0)
            y = max(col * tile_w - overlap_w, 0)
            # if it exceeds horizontally or vertically in the last rows or cols, increase overlap to fit
            if (x + tile_h) >= height:
                x -= abs(x + tile_h - height)
            if (y + tile_w) >= width:
                y -= abs(y + tile_w - width)
            # assign tile to final tensor
            tiles[row, col] = image[x:x + tile_h, y:y + tile_w, :]
    return tiles


def window_crop(raster: DatasetReader,
                polygons: List[Polygon],
                min_size: int = 256,
                padding: Union[tuple, int] = 16) -> win.Window:
    """Generates a window crop from the given raster, making sure than each side is at least `min_size` pixels wide.
    Padding also applies and it is included in the window size before the minimum requirement check.

    Args:
        raster (DatasetReader): input rasterio Dataset
        polygons (List[Polygon]): list of Shapely polygons
        min_size (int, optional): Minimum pixel size of each dimension. Defaults to 256.
        padding (Union[tuple, int], optional): How much to pad the window, specify a tuple for horizontal and vertical. Defaults to 16.

    Returns:
        Window: rasterio Window
    """
    padx, pady = padding if isinstance(padding, tuple) else (padding, padding)
    window = features.geometry_window(raster, polygons, pad_x=padx, pad_y=pady)

    h = max(window.height, min_size)
    w = max(window.width, min_size)
    if h > window.height or w > window.width:
        centroid = MultiPolygon(polygons).centroid
        px, py = raster.index(centroid.x, centroid.y)
        window = win.Window(py - np.ceil(h / 2), px - np.ceil(w / 2), h, w)
    return window


def store_window(source: DatasetReader, destination: Union[Any, MemoryFile], window: win.Window) -> None:
    """Saves the given window, obtained from the input source, to the destination file.
    This is a simple wrapper around rasterio functionalities.

    Args:
        source (DatasetReader): input dataset source, as derived from rasterio.open()
        destination (Union[Any, MemoryFile]): generic file, MemoryFile or whatever has an "open" function
        window (win.Window): rasterio Window object, defining a subregion of source
    """
    metadata = source.meta.copy()
    metadata.update({
        "height": window.height,
        "width": window.width,
        "transform": win.transform(window, source.transform)
    })
    with destination.open(**metadata) as dataset:
        dataset.write(source.read(window=window))
