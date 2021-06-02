import numpy as np
import cv2
import math
import warnings
from copy import copy

warnings.filterwarnings('ignore')
DEBUG_SIZE_PX = [500, 500]

class Image:

    """
    Top-level class for managing image operations. By default operations are
    done in-place but return an updated instance for further manipulations.
    """

    class Decorators:

        @classmethod
        def supply_new(decs, func):
            """
            Perform the specified op on a new (copy) Image object
            """
            def inner(cls, *args, **kwargs):
                out = copy(cls)
                out.arr = cls.arr.copy()
                cls = func(out, *args, **kwargs)
                return cls
            return inner

        @classmethod
        def prep_8UC1(decs, func):
            """
            Supply a normalized *greyscale* version of the image.
            """
            @decs.supply_new
            def inner(cls, *args, **kwargs):
                cls.cvt8UC1()
                out = func(cls, *args, **kwargs)
                return out
            return inner

        @classmethod
        def reset_dimensions(decs, func):
            """
            Get a new height/width value after any relevant ops.
            """
            def inner(cls, *args, **kwargs):
                out = func(cls, *args, **kwargs)   
                out.h, out.w = out.arr.shape[:2] 
                return out
            return inner     
                
    def __init__(self, in_obj):
        """
        Initialise image as numpy array with h/w and other info.
        """
        if isinstance(in_obj, str):  # If str assume path to saved .npy array
            self.arr = np.load(in_obj).astype('float64')
        else:
            # Otherwise assume live array - copy for hygiene
            self.arr = in_obj.copy()
        
        self.h, self.w = self.arr.shape[:2]

        if self.arr.shape != [self.h, self.w]:
            self.arr = self.arr.reshape(self.h, self.w)  # Force into shape

    @Decorators.prep_8UC1
    def adaptive_threshold(self, block_size, c_shift):

        """Do cv2 adaptive thresholding using parameters specified"""

        self.arr = cv2.adaptiveThreshold(self.arr, 
                                         255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         block_size,
                                         c_shift)
        return self

    def affine_transform(self, a, x_off, y_off):
        """
        Build and apply affine transformation matrix from component parts.
        """
        M = cv2.getRotationMatrix2D((int(self.w / 2), int(self.h / 2)), a, 1)
        
        M[0, 2] += x_off
        M[1, 2] += y_off

        self.arr = cv2.warpAffine(self.arr, M, (self.w, self.h))

        return self

    def cast(self, casttype):
        """
        Cast the image array to the type specified.
        """
        self.arr = self.arr.astype(casttype)
        
        return self

    @Decorators.reset_dimensions
    def crop(self, pt0, pt1):
        """
        Do straightforward rectangular cropping from pt --> pt.
        """        
        x_lo, y_hi = [x if x > 0 else 0 for x in pt0]
        x_hi, y_lo = [x if x > 0 else 0 for x in pt1]

        self.arr = self.arr[y_hi:y_lo, x_lo:x_hi]

        return self

    def cut_box(self, x, y, w, h, pad=0):
        """
        Cut out a box and return as a new image object.
        """
        pt0 = (x - pad, y - pad)
        pt1 = (x + w + pad, y + h + pad)

        self.crop(pt0, pt1)

        # Return any overlap which was missed by the crop
        off_h = -(x - pad) if x - pad < 0 else 0
        off_v = -(y - pad) if y - pad < 0 else 0
        
        return self, (off_h, off_v)

    def cvt8UC1(self):
        """
        Helper function for utility functions which need this format but don't
        return the image.
        """
        self.norm_minmax().cast('uint8')
        
        return self

    def cvt2color(self):
        """
        Convert a greyscale image to a color image.
        """
        assert len(self.arr.shape) == 2, Exception('Too many channels')

        self.arr = cv2.merge([self.arr, self.arr, self.arr])

        return self

    def draw_boxes(self, boxes, fill_type=1, color=(0, 0, 255)):
        """
        Draw the boxes onto the image using default settings.
        """
        for x, y, w, h in boxes:
            cv2.rectangle(self.arr, (x, y), (x + w, y + h), color, fill_type)
        return self

    def draw_contours(self, cnts, fill_type=1, color=(0, 0, 255)):
        """
        Draw the contours onto the image using default settings.
        """
        cv2.drawContours(self.arr, cnts, -1, color, fill_type)
        
        return self

    def draw_lines(self, lines, fill_type=1, color=(0, 0, 255)):
        """
        Draw lines onto an image.
        """
        for pt1, pt2 in lines:
            cv2.line(self.arr, pt1, pt2, color, fill_type, cv2.LINE_AA)

    def draw_polys(self, polys, fill_type=1, color=(0, 0, 255)):
        """
        Draw a (closed) polygon onto an image.
        """
        for pts in polys:
            lines = []
            for i, pt0 in enumerate(pts):
                pt1 = pts[i+1] if i < len(pts) - 1 else pts[0]
                lines.append((pt0[0], pt1[0]))
            self.draw_lines(lines, color=color) 

        return self

    def flood_fill(self, color):
        """
        Fill contained spaces from the inside out.
        """
        mask = np.zeros((self.h + 2, self.w + 2), np.uint8)
        cv2.floodFill(self.arr, mask, [int(x) for x in self.c], color)

        return self

    ####   'get_' methods return something other than the image itself   ####

    @Decorators.prep_8UC1
    def get_contours(self, as_mbrs=False):
        """
        Shorthand for the cv2 findContours method.
        """
        cnts, _ = cv2.findContours(self.arr, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        
        # Return in descending order of size (area)
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))[::-1]

        if as_mbrs:
            for i, cnt in enumerate(cnts):
                cnt = cnt.reshape(len(cnt), 2)
                pt0 = min([x[0] for x in cnt]), min([x[1] for x in cnt])
                pt1 = max([x[0] for x in cnt]), max([x[1] for x in cnt])

                cnts[i] = (*pt0, pt1[0] - pt0[0], pt1[1] - pt0[1])

        return cnts

    @Decorators.supply_new
    def get_copy(self):
        """
        Copy existing class applying hard copy to array attribute.
        """
        return self

    def get_extremities(self):
        """
        Get the outermost points in every direction of nonzero pixels.
        """
        hi, lo = self.nonzero[0][0], self.nonzero[0][-1]
        l, r   = self.T.nonzero[0][0], self.T.nonzero[0][-1]

        return hi, lo, l, r

    @Decorators.prep_8UC1
    def get_hough_lines(self, thresh, min_len, max_gap):
        """
        Return the result of a hough transform when applied to the image.
        """
        lines = cv2.HoughLinesP(self.arr, 1, np.pi / 180, thresh, None,
                                                            min_len, max_gap)
        if len(lines) == 0:
            return []
        else:
            return lines.reshape(len(lines), 2, 2)  # Removes additional dim

    @Decorators.supply_new
    def get_template(self, color=0):
        """
        Return Image object with zero (black) array of same size.
        """
        if color == 0:
            self.arr = np.zeros_like(self.arr).astype('uint8')
        else:
            self.arr = np.full([self.h, self.w], color).astype('uint8')
        
        return self

    def morph(self, morph_type, kernel):
        """
        Shorthand for cv2 morphologyEx method - requires kernel/type spec.
        """
        op = eval('cv2.MORPH_' + morph_type.upper())
        self.arr = cv2.morphologyEx(self.arr, op, kernel, cv2.BORDER_DEFAULT)

        return self

    def norm_minmax(self, lo=None, hi=None):
        """
        Normalize such that the specified values are set to 0/255.
        """
        hi = hi if hi else self.hi
        lo = lo if lo else self.lo
        
        self.arr = self.arr - lo

        scale_hi = 255 / (hi - lo)
        self.arr = self.arr * scale_hi

        return self

    @Decorators.reset_dimensions
    def pad(self, hi, lo, l, r, color=0):
        """
        Add black padding to the image as specified.
        """
        pad_hi = np.full([hi, self.w], color)
        pad_lo = np.full([lo, self.w], color)

        self.arr = np.vstack([pad_hi, self.arr, pad_lo])

        pad_l = np.full([self.h + hi + lo, l], color)
        pad_r = np.full([self.h + hi + lo, r], color)
        
        self.arr = np.hstack([pad_l, self.arr, pad_r])

        return self

    def put_text(self, text):
        """
        Shorthand for cv2 puttext method.
        """
        cv2.putText(self.arr, str(text), (2, 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.25, (0, 0, 255), 1)

        return self

    @Decorators.reset_dimensions
    def resize(self, w=None, h=None):
        """
        Resize the image to the dimensions specified.
        """
        w = w if w else self.w
        h = h if h else self.h

        self.arr = cv2.resize(self.arr, (w, h))

        return self

    def rotate(self, theta):
        """
        Spin the image through the angle specified.
        """
        M = cv2.getRotationMatrix2D(self.c, theta, 1)
        self.arr = cv2.warpAffine(self.arr, M, (self.w, self.h))

        return self

    def sharpen_edges(self, hi_k, lo_k, thresh=True):
        """
        Erode using specified kernel to get sharper *white* edges.
        """
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
        self.cvt2color().draw_contours(cnts).draw_boxes(boxes).draw_lines(lines)
        self.put_text(text).draw_polys(polys)
        
        self.resize(*DEBUG_SIZE_PX)
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

    ####   Useful properties are evaluated on-demand   #####

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

    @property
    def nonzero(self):
        return np.nonzero(self.arr)

    @property
    def T(self):
        return Image(self.arr.T)