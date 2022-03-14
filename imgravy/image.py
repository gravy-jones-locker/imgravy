from __future__ import annotations

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
        def supply_new(decs: Decorators, func: function):
            """Perform the specified op on a new (copy) Image object"""
            def inner(im: Image, *args, **kwargs):
                return func(type(im)(im.arr.copy()), *args, **kwargs)
            return inner

        @classmethod
        def prep_8UC1(decs: Decorators,  func: function):
            """Supply a rounded version of the image with appropriate dtype"""
            @decs.supply_new
            def inner(im: Image, *args, **kwargs):
                im.cvt8UC1()
                return func(im, *args, **kwargs)
            return inner
        
        @classmethod
        def return_res(decs: Decorators,  func: function):
            """Replies the altered Image rather than function output"""
            def inner(im: Image, *args, **kwargs):
                func(im, *args, **kwargs)
                return im
            return inner

    def __init__(self, in_arr: np.array):
        self.arr = in_arr.copy()  # Bind copy for hygiene
        
    @classmethod
    def _from_bytes(cls, stream: bytes) -> Image:
        """Load an Image class from a bytes stream"""
        return cls(cv2.imdecode(np.frombuffer(stream, np.uint8), flags=1))
    
    def _to_bytes(self, ftype: str) -> bytes:
        """Convert an image array to a stream of bytes"""
        return cv2.imencode(ftype, self.arr)[1].tobytes()

    @Decorators.return_res
    @Decorators.prep_8UC1
    def adaptive_threshold(self, block_size: float, c_shift: float) -> None:
        """Do cv2 adaptive thresholding using parameters specified"""
        self.arr = cv2.adaptiveThreshold(self.arr, 
                                         255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
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
    def cast(self, casttype: str) -> None:
        """Cast the image array to the type specified"""
        self.arr = self.arr.astype(casttype)

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
        self.norm_minmax().cast('uint8')

    @Decorators.return_res
    def cvt2color(self) -> None:
        """Convert a greyscale image to a color image"""
        if len(self.arr.shape) == 2:
            raise Exception('Can\'t convert from color -> color')
        self.arr = cv2.merge([self.arr, self.arr, self.arr])

    @Decorators.return_res
    def draw_boxes(self, boxes: list, fill_type: int=1, 
    color: tuple=(0, 0, 255)) -> None:
        """Draw boxes onto an image"""
        for x, y, w, h in boxes:
            cv2.rectangle(self.arr, (x, y), (x + w, y + h), color, fill_type)

    @Decorators.return_res
    def draw_contours(self, cnts: list, fill_type: int=1,
    color: tuple=(0, 0, 255)) -> None:
        """Draw contours onto an image"""
        cv2.drawContours(self.arr, cnts, -1, color, fill_type)

    @Decorators.return_res
    def draw_lines(self, lines: list, fill_type: int=1,
    color: tuple=(0, 0, 255)) -> None:
        """Draw lines onto an image"""
        for pt1, pt2 in lines:
            cv2.line(self.arr, pt1, pt2, color, fill_type, cv2.LINE_AA)

    @Decorators.return_res
    def draw_polys(self, polys: list, fill_type: int=1,
    color: tuple=(0, 0, 255)) -> None:
        """Draw a (closed) polygon onto an image"""
        for pts in polys:
            lines = []
            for i, pt0 in enumerate(pts):
                pt1 = pts[i+1] if i < len(pts) - 1 else pts[0]
                lines.append((pt0[0], pt1[0]))
            self.draw_lines(lines, color=color) 

    @Decorators.return_res  # TODO I don't know what this does
    def flood_fill(self, c: list, color: tuple=(0, 0, 255)) -> None:
        """Fill contained spaces from the inside out"""
        mask = np.zeros((self.h + 2, self.w + 2), np.uint8)
        cv2.floodFill(self.arr, mask, [int(x) for x in c], color)

    ####   'get_' methods return something other than the image itself   ####

    @Decorators.prep_8UC1
    def get_contours(self, as_mbrs: bool=False) -> list:
        """Shorthand for the cv2 findContours method"""
        cnts, _ = cv2.findContours(self.arr,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnts.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        if as_mbrs:
            for i, cnt in enumerate(cnts):
                cnts[i] = cv2.boxPoints(cv2.minAreaRect(cnt))
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

    @Decorators.return_res
    @Decorators.supply_new
    def get_template(self, color: Union[int, tuple]=0) -> None:
        """Return new Image with array of same size and block color"""
        self.arr = np.full([self.h, self.w], color).astype('uint8')

    @Decorators.return_res
    def morph(self, morph_type: str, kernel: np.array) -> None:
        """Shorthand for cv2 morphologyEx method - requires kernel/type spec"""
        self.arr = cv2.morphologyEx(self.arr,
                                    eval('cv2.MORPH_' + morph_type.upper()),
                                    kernel,
                                    cv2.BORDER_DEFAULT)

    @Decorators.return_res
    def norm_minmax(self, lo: int=None, hi: int=None) -> None:
        """Normalize such that the specified values are set to 0/255"""
        if hi is None:
            hi = self.hi  # Set min/max to limits of image if None passed
        if lo is None:
            lo = self.lo
        self.arr = self.arr - lo
        self.arr = self.arr * 255 / (hi - lo)

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
    def put_text(self, text: str, size: int, 
    color: Union[int, tuple]=255) -> None:
        """Shorthand for cv2 puttext method"""
        cv2.putText(
        self.arr, str(text), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)

    @Decorators.return_res
    def resize(self, new_w: int=None, new_h: int=None) -> None:
        """Resize the image to the dimensions specified"""
        self.arr = cv2.resize(self.arr, (new_w if new_w else self.w, 
                                         new_h if new_h else self.h))

    def rotate(self, theta: float, bound: bool=True, bd: int=0) -> None:
        """Spin the image through the angle specified"""
        M = cv2.getRotationMatrix2D(self.c, theta, 1)
        if bound:
            # TODO I'm not sure what this does either
            cos, sin = np.abs(M[0])
            
            w = int((self.h * sin) + (self.w * cos))
            h = int((self.h * cos) + (self.w * sin))

            off_x = int((w / 2) - self.c[0])
            off_y = int((h / 2) - self.c[1])

            M[0, 2] += off_x
            M[1, 2] += off_y
        else:
            w, h = self.w, self.h
            off_x, off_y = 0, 0

        self.arr = cv2.warpAffine(self.arr, M, (w, h), borderValue=bd)

        return self, (off_x, off_y)

    def sharpen_edges(self, hi_k, lo_k, thresh=True):
        """Erode using specified kernel to get sharper *white* edges"""
        v_kernel = np.ones([hi_k, lo_k])
        self.arr = cv2.erode(self.arr, v_kernel)
        self.arr = cv2.dilate(self.arr, v_kernel)

        h_kernel = np.ones([lo_k, hi_k])
        self.arr = cv2.erode(self.arr, h_kernel)
        self.arr = cv2.dilate(self.arr, h_kernel)

        if thresh:
            self.threshold(1)
        
        return self

    def slice_rng(self, lo=None, hi=None):
        """
        Slice the intensities in the image to those inside the range specified.
        """
        if lo:
            self.arr[self.arr < lo] = self.lo
        if hi:
            self.arr[self.arr > hi] = self.hi

        return self

    @Decorators.prep_8UC1
    def show(self, cnts=[], boxes=[], text='', lines=[], polys=[]):
        """
        Normalize and show the image for debugging purposes.
        """
        # Operations will have no effect if no objects supplied
        if len(self.arr.shape) == 2:
            self.cvt2color()
        self.draw_contours(cnts).draw_boxes(boxes).draw_lines(lines)
        self.put_text(text).draw_polys(polys)

        if self.h > self.w:
            ratio = DEBUG_SIZE_PX / self.h
            DEBUG_DIMS = int(self.w * ratio), DEBUG_SIZE_PX
        else:
            ratio = DEBUG_SIZE_PX / self.w
            DEBUG_DIMS = DEBUG_SIZE_PX, int(self.h * ratio)
        
        self.resize(*DEBUG_DIMS)
        cv2.imshow('progress', self.arr)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()

    def threshold(self, thresh_val):
        """
        Apply binary thresholding using the value specified.
        """
        _, self.arr = cv2.threshold(self.arr, thresh_val, 255,
                                                            cv2.THRESH_BINARY)
        return self
    
    ####   Expensive properties evaluated as private functions   ####

    def _nonzero(self):
        return np.nonzero(self.arr)

    def _T(self):
        return Image(self.arr.T)

    ####   Useful properties are evaluated on-demand   #####

    @property
    def h(self):
        return self.size[0]
    
    @property
    def w(self):
        return self.size[1]

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
