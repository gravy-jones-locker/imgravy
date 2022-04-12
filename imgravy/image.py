from __future__ import annotations

from typing import Union
from scipy import stats

import numpy as np
import cv2

DEBUG_SIZE_PX = 900

class Image:
    """
    Top-level class for managing image operations. By default operations are
    done in-place but return an updated instance for further manipulations.
    """
    class Decorators:

        @classmethod
        def supply_new(decs, func: function):
            """Perform the specified op on a new (copy) Image object"""
            def inner(im: Image, *args, **kwargs):
                return func(type(im)(im.arr.copy()), *args, **kwargs)
            return inner

        @classmethod
        def prep_8UC1(decs,  func: function):
            """Supply a rounded version of the image with appropriate dtype"""
            def inner(im: Image, *args, **kwargs):
                im.cvt8UC1()
                return func(im, *args, **kwargs)
            return inner
        
        @classmethod
        def return_res(decs,  func: function):
            """Returns the altered Image rather than function output"""
            def inner(im: Image, *args, **kwargs):
                func(im, *args, **kwargs)
                return im
            return inner

    def __init__(self, in_obj: Union[str, bytes, np.ndarray],
    grayscale: bool=False) -> None:
        """Execute correct method to bind array from the input object"""
        self.arr = self._parse_in_obj(in_obj)
        if grayscale:
            self.cvt2gray()
    
    def _parse_in_obj(self, in_obj: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Get array from generic input object"""
        if isinstance(in_obj, np.ndarray):
            return in_obj
        if isinstance(in_obj, str):
            return self._parse_file(in_obj)
        if isinstance(in_obj, bytes):
            return self._parse_bytes(in_obj)
        
    def _parse_bytes(self, stream: bytes) -> Image:
        """Load an Image class from a bytes stream"""
        return cv2.imdecode(np.frombuffer(stream, np.uint8), flags=1)
    
    def _parse_file(self, fpath: str) -> Image:
        """Load an Image class from a file path"""
        return cv2.imread(fpath)
    
    def _to_bytes(self, ftype: str) -> bytes:
        """Convert an image array to a stream of bytes"""
        return cv2.imencode(ftype, self.arr)[1].tobytes()

    @Decorators.return_res
    @Decorators.prep_8UC1
    def adaptive_threshold(self, block_size: float, c_shift: float) -> None:
        """Do cv2 adaptive thresholding using parameters specified"""
        self.arr = cv2.adaptiveThreshold(self.arr, 
                                         255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY,
                                         block_size,
                                         c_shift)

    @Decorators.return_res
    def affine_trans(self, theta: float, x_off: float, y_off: float) -> None:
        """Build and apply affine transformation matrix from component parts"""
        M = cv2.getRotationMatrix2D(self.c, theta, 1)
        M[0, 2] += x_off
        M[1, 2] += y_off
        self.arr = cv2.warpAffine(self.arr, M, (self.w, self.h))
    
    @Decorators.return_res
    def annotate(self, **annotations) -> None:
        """Draws on boxes/lines/contours/text/polys as specified"""
        if len(self.arr.shape) == 2:
            self.cvt2color()  # Guarantees annotations will show
        for key, value in annotations.items():
            eval(f'self.draw_{key}(value)')
    
    @Decorators.return_res
    def blur_gaussian(self, kernel: tuple) -> None:
        """Do Gaussian blurring"""
        self.arr = cv2.GaussianBlur(self.arr, kernel, 0)

    @Decorators.return_res
    def cast(self, casttype: str) -> None:
        """Cast the image array to the type specified"""
        self.arr = self.arr.astype(casttype)
    
    def copy(self) -> Image:
        """Provide a copy of the Image object for unconnected edits"""
        return type(self)(self.arr.copy())

    @Decorators.return_res
    def crop(self, pt0: tuple, pt1: tuple) -> None:
        """Do straightforward rectangular cropping from pt --> pt"""        
        x_lo, y_hi = [x if x > 0 else 0 for x in pt0]  # Replace any sub-0 vals
        x_hi, y_lo = [x if x > 0 else 0 for x in pt1]

        self.arr = self.arr[y_hi:y_lo, x_lo:x_hi]

    def cut_box(self, x: int, y: int, w: int, h: int, pad: int=0) -> tuple:
        """Cut out a box and return as a new image object"""
        self.crop((x - pad, y - pad), (x + w + pad, y + h + pad))

        # Return any overlap which was missed by the crop
        off_h = pad - x if x - pad < 0 else 0
        off_v = pad - y if y - pad < 0 else 0
        
        return self, (off_h, off_v)

    @Decorators.return_res
    def cvt8UC1(self) -> None:
        """Converts into filter-friendly format (i.e. unsigned 0-255)"""
        self.cast('uint8')

    @Decorators.return_res
    def cvt2color(self) -> None:
        """Convert a greyscale image to a color image"""
        if len(self.arr.shape) == 3:
            return  # True if already color
        self.arr = cv2.merge([self.arr, self.arr, self.arr])
    
    @Decorators.return_res
    def cvt2gray(self) -> None:
        """Convert a color image to a grayscale image"""
        if len(self.arr.shape) == 2:
            return  # True if already grayscale
        self.arr = cv2.cvtColor(self.arr, cv2.COLOR_BGR2GRAY)
    
    @Decorators.return_res
    def cvt2hsv(self) -> None:
        """Convert from BGR colorspace to HSV colorspace"""
        self.arr = cv2.cvtColor(self.arr, cv2.COLOR_BGR2HSV)

    @Decorators.return_res
    def draw_boxes(self, boxes: list, fill_type: int=3, 
    color: tuple=(0, 255, 0)) -> None:
        """Draw boxes onto an image"""
        for x, y, w, h in boxes:
            cv2.rectangle(self.arr, (x, y), (x + w, y + h), color, fill_type)

    @Decorators.return_res
    def draw_contours(self, cnts: list, fill_type: int=3,
    color: tuple=(0, 255, 0)) -> None:
        """Draw contours onto an image"""
        cv2.drawContours(self.arr, cnts, -1, color, fill_type)

    @Decorators.return_res
    def draw_lines(self, lines: list, width: int=1,
    color: tuple=(0, 0, 255)) -> None:
        """Draw lines onto an image"""
        for pt1, pt2 in lines:
            cv2.line(self.arr, pt1, pt2, color, width, cv2.LINE_AA)

    @Decorators.return_res
    def draw_polys(self, polys: list, color: tuple=(0, 0, 255)) -> None:
        """Draw a (closed) polygon onto an image"""
        for pts in polys:
            lines = []
            for i, pt0 in enumerate(pts):
                pt1 = pts[i+1] if i < len(pts) - 1 else pts[0]
                lines.append((pt0[0], pt1[0]))
            self.draw_lines(lines, color=color) 

    @Decorators.return_res
    def draw_text(self, text: str, position: tuple, size: int, 
    color: Union[int, tuple]=255) -> None:
        """Shorthand for cv2 putText method"""
        cv2.putText(self.arr,
                    str(text),
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size,
                    color,
                    1)
    
    @Decorators.return_res
    def filter_bilateral(self, d: int, sig_color: int, sig_space: int) -> None:
        """Do bilateral filtering with the arguments passed"""
        self.arr = cv2.bilateralFilter(self.arr, d, sig_color, sig_space)
    
    @Decorators.return_res
    def find_diff(self, smooth_k: int, blur_k: int) -> None:
        """Subtract image from its average to expose edges"""
        self.arr = cv2.GaussianBlur(self.arr, (smooth_k, smooth_k), 0)
        blur = cv2.GaussianBlur(self.arr, (blur_k, blur_k), 0)
        self.arr = blur - self.arr
        self.arr[self.arr > 255] = 255
        self.arr[self.arr < 0] = 0
    
    @Decorators.return_res
    def fit_window(self) -> None:
        """Resize to the max dim given in DEBUG_SIZE_PX"""
        if self.h > self.w:
            ratio = DEBUG_SIZE_PX / self.h
            DEBUG_DIMS = int(self.w * ratio), DEBUG_SIZE_PX
        else:
            ratio = DEBUG_SIZE_PX / self.w
            DEBUG_DIMS = DEBUG_SIZE_PX, int(self.h * ratio)
        self.resize(*DEBUG_DIMS)

    @Decorators.return_res  # TODO I don't know what this does
    def flood_fill(self, c: list, color: tuple=(0, 0, 255)) -> None:
        """Fill contained spaces from the inside out"""
        mask = np.zeros((self.h + 2, self.w + 2), np.uint8)
        cv2.floodFill(self.arr, mask, [int(x) for x in c], color)

    ####   'get_' methods return something other than the image itself   ####

    def get_channel_px(self, i: int, flatten: bool=False) -> np.ndarray:
        """Return an array of pixels in the channel at i"""
        out = self.arr[..., i]
        if flatten:
            out = out.flatten()
        return out

    @Decorators.prep_8UC1
    def get_contours(self, as_mbrs: bool=False) -> list:
        """Shorthand for the cv2 findContours method"""
        cnts = list(cv2.findContours(self.arr,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[0])
        cnts.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        if as_mbrs:
            for i, cnt in enumerate(cnts):
                cnts[i] = cv2.boundingRect(cnt)
        return cnts

    def get_extremities(self):
        """Get the outermost points in every direction of nonzero pixels"""
        y_pts, x_pts = self._nonzero()[:2]
        hi, lo = np.min(y_pts), np.max(y_pts)
        l, r   = np.min(x_pts), np.max(x_pts)
        return hi, lo, l, r

    @Decorators.prep_8UC1
    def get_hough_lines(self, thresh: float, min_len: float,
    max_gap: float) -> list:
        """Return the result of a hough transform"""
        lines = cv2.HoughLinesP(
        self.arr, 1, np.pi / 180, thresh, None, min_len, max_gap)
        if lines is None:
            return []
        return lines.reshape(len(lines), 2, 2)  # Removes additional axis
    
    def get_sample(self, dec: float) -> np.ndarray:
        """Get a sample of the specified decimal"""
        sample_size = int(self.arr.size * dec / 3)
        return self.arr[np.random.randint(self.h, size=sample_size),
                        np.random.randint(self.w, size=sample_size)]

    @Decorators.supply_new
    @Decorators.return_res
    def get_template(self, color: Union[int, tuple]=0) -> None:
        """Return new Image with array of same size and block color"""
        self.arr = np.full([self.h, self.w], color).astype('uint8')

    @Decorators.return_res
    def invert(self):
        """Invert the values such that 0/255 swap"""
        self.arr = 255 - self.arr

    @Decorators.return_res
    def morph(self, morph_type: str, kernel: np.array) -> None:
        """Shorthand for cv2 morphologyEx method - requires kernel/type spec"""
        self.arr = cv2.morphologyEx(self.arr,
                                    eval('cv2.MORPH_' + morph_type.upper()),
                                    kernel,
                                    cv2.BORDER_DEFAULT)

    @Decorators.return_res
    def norm_minmax(self, lo: int=None, hi: int=None, quantiles: tuple=None,
    bound: bool=False, keep_dtype: bool=False) -> None:
        """Normalize such that the specified values are set to 0/255"""
        lo, hi = self._parse_norm_limits(lo, hi, quantiles)
        dtype = self.arr.dtype
        self.arr = (self.cast('float64').arr - lo) * (255 / (hi - lo))
        if bound:
            self.arr[self.arr > 255] = 255
            self.arr[self.arr < 0] = 0
        if keep_dtype and dtype != self.arr.dtype:
            self.arr = self.cast(dtype)

    @Decorators.return_res
    def pad(self, hi: int, lo: int, l: int, r: int,
    color: Union[int, tuple]=0) -> None:
        """Add padding to the image as specified"""
        def build(*dims):
            return np.full(dims, color, dtype=self.arr.dtype)
        self.arr = np.vstack([build(hi, self.w), self.arr, build(lo, self.w)])
        self.arr = np.hstack([build(self.h, l), self.arr, build(self.h, r)])

    def save(self, fpath: str) -> None:
        """Save to the location specified"""
        cv2.imwrite(fpath, self.arr)
    
    @Decorators.return_res
    def select_channels(self, channels: list) -> None:
        """Selects the specified channels only"""
        self.arr = self.arr[..., channels]
    
    @Decorators.return_res
    def reduce_bitrate(self) -> None:
        """Convert to four bit image"""
        self.arr = ((self.arr.astype(int) >> 4) << 4) + ((1 << 4) >> 1)

    @Decorators.return_res
    def resize(self, new_w: int=None, new_h: int=None) -> None:
        """Resize the image to the dimensions specified"""
        self.arr = cv2.resize(self.arr, (new_w if new_w else self.w, 
                                         new_h if new_h else self.h))

    def rotate(self, theta: float, bound: bool=True, bd: int=0) -> None:
        """Spin the image through the angle specified"""
        M = cv2.getRotationMatrix2D(self.c, theta, 1)
        w = self.w
        h = self.h
        off_x, off_y = 0, 0
        if bound:  # Maintains absolute size in new angle of rotation
            cos, sin = np.abs(M[0, :2])
            w = int((self.h * sin) + (self.w * cos))
            h = int((self.h * cos) + (self.w * sin))
            off_x = int((w / 2) - self.c[0])
            off_y = int((h / 2) - self.c[1])
            M[0, 2] += off_x
            M[1, 2] += off_y
        self.arr = cv2.warpAffine(self.arr, M, (w, h), borderValue=bd)
        return self, (off_x, off_y)

    @Decorators.return_res
    def sharpen_edges(self, k_dims: tuple, thresh: bool=True):
        """Erode using specified kernel to get sharper *white* edges"""
        for dims in k_dims, k_dims[::-1]:
            kernel = np.ones(dims)
            # Erode then dilate to extract continuous horiz/vert lines
            for method in cv2.erode, cv2.dilate:
                self.arr = method(self.arr, kernel)
        if thresh:
            self.threshold(1)

    @Decorators.return_res
    def slice_rng(self, lo: float=None, hi: float=None) -> None:
        """Slice the intensities in the image to those inside the range"""
        if lo is not None:
            self.arr[self.arr < lo] = self.lo
        if hi is not None:
            self.arr[self.arr > hi] = self.hi

    @Decorators.supply_new
    @Decorators.prep_8UC1
    def show(self, resize: bool=True, **annotations) -> None:
        """Annotate/normalize the image"""
        self.annotate(**annotations)
        if resize:
            self.fit_window()
        cv2.imshow('progress', self.arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @Decorators.return_res
    def threshold(self, thresh_val):
        """
        Apply binary thresholding using the value specified.
        """
        _, self.arr = cv2.threshold(self.arr,
                                    thresh_val,
                                    255,
                                    cv2.THRESH_BINARY)
    
    ####   Expensive properties evaluated as private functions   ####

    def _nonzero(self):
        return np.nonzero(self.arr)

    def _T(self):
        return Image(self.arr.T)
    
    def _quantile(self, q: float) -> float:
        return np.quantile(self.arr, q)
    
    def _mode(self) -> np.ndarray:
        out = []
        for i in range(self.arr.shape[-1]):
            channel_px = self.get_channel_px(i, flatten=True)
            channel_mode = stats.mode(channel_px, axis=None)[0][0]
            out.append(channel_mode)
        return np.array(out)

    ####   Useful properties are evaluated on-demand   #####

    @property
    def h(self):
        return self.arr.shape[0]
    
    @property
    def w(self):
        return self.arr.shape[1]

    @property
    def area(self):
        return self.h * self.w

    @property
    def hi(self):
        return np.max(self.arr)

    @property
    def lo(self):
        return np.min(self.arr)

    @property
    def c(self):
        return self.w / 2, self.h / 2

    def _parse_norm_limits(self, lo: int, hi: int, quantiles: tuple) -> tuple:
        if quantiles is not None:
            return (np.quantile(self.arr, q) for q in quantiles)
        return self.lo if lo is None else lo, self.hi if hi is None else hi