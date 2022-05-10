import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
import soundfile

import cv2
import math

from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

# class AudioVisualDataset(collections.abc.Mapping):
#     def __init__(self, path, dtype=None):
#         assert check_argument_types()
#         self.loader = path
#         self.dtype = dtype
#         self.rate = None

#     def keys(self):
#         return self.loader.keys()

#     def __len__(self):
#         return len(self.loader)

#     def __iter__(self):
#         return iter(self.loader)

#     def __getitem__(self, key: str) -> np.ndarray:
#         retval = self.loader[key]

#         if isinstance(retval, tuple):
#             assert len(retval) == 2, len(retval)
#             if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
#                 # sound scp case
#                 rate, array = retval
#             elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
#                 # Extended ark format case
#                 array, rate = retval
#             else:
#                 raise RuntimeError(
#                     f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
#                 )

#             if self.rate is not None and self.rate != rate:
#                 raise RuntimeError(
#                     f"Sampling rates are mismatched: {self.rate} != {rate}"
#                 )
#             self.rate = rate
#             # Multichannel wave fie
#             # array: (NSample, Channel) or (Nsample)
#             if self.dtype is not None:
#                 array = array.astype(self.dtype)

#         else:
#             # Normal ark case
#             assert isinstance(retval, np.ndarray), type(retval)
#             array = retval
#             if self.dtype is not None:
#                 array = array.astype(self.dtype)

#         assert isinstance(array, np.ndarray), type(array)
#         return array


class VisionFileReader(collections.abc.Mapping):
    """Reader class for 'vision'.

    Examples:
        key1 /some/path/a.mp4
        key2 /some/path/b.mp4
        key3 /some/path/c.mp4
        key4 /some/path/d.mp4
        ...

        >>> reader = VisionFileReader('vision')
        >>> rate, array = reader['key1']
    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
        normalize: bool = False,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.normalize = normalize
        self.data = read_2column_text(fname)

    def __getitem__(self, key):
        # Returns a cv2 video capture instance
        mp4 = self.data[key]
        vid = cv2.VideoCapture(mp4)
        rate = vid.get(cv2.CAP_PROP_FPS) #fps
        return rate, vid
    
    def get_normalize(self):
        return self.normalize

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class VisionDataset(collections.abc.Mapping):
    def __init__(self, loader, wav_rate = None, vis_step = None, dtype=None):
        assert check_argument_types()
        self.loader = loader
        self.dtype = dtype
        self.rate = wav_rate
        self.vis_step = vis_step
        self.normalize = loader.get_normalize()
        self.vid_rate = 0
        self.sample_rate = 0
        self.sample_step = 0
        self.set_sample_rate()

    def set_sample_rate(self):
           
        retval = self.loader[list(self.keys())[0]]
        assert len(retval) == 2, len(retval)
        rate, vidcap = retval
        if self.rate is not None and self.rate != rate:
            vidcap.set(cv2.CAP_PROP_FPS, self.rate)
        # Multichannel mp4 file

        self.vid_rate = vidcap.get(cv2.CAP_PROP_FPS)
        min_vis_step = self.rate // self.vid_rate
        sample_vis_step = max(min_vis_step, self.vis_step)
        self.sample_step = int(math.ceil(sample_vis_step / min_vis_step))
        self.vis_step = self.sample_step * min_vis_step
        self.sample_rate = self.vid_rate / self.sample_step


    def keys(self):
        return self.loader.keys()
    
    def get_vid_rate(self):
        return self.vid_rate
    
    def get_sample_rate(self):
        return self.sample_rate

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]
        
        assert len(retval) == 2, len(retval)
        rate, vid = retval

        if self.rate is not None and self.rate != rate:
            vid.set(cv2.CAP_PROP_FPS, self.rate)
        # Multichannel mp4 file

        array = self.capture_video(vid)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        assert isinstance(array, np.ndarray), type(array)
        return array

    def capture_video(self, vidcap):
        # How to get a image capture in a specific time stamp for video
        data = []
        
        count = 0
        total_length = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        assert(self.vid_rate == vidcap.get(cv2.CAP_PROP_FPS))
        success = True
        success, first_image = vidcap.read()
        num_captures = 0
        cap_shape = first_image.shape
        while success:
            success,image = vidcap.read()
            if count % self.sample_step == 0:
                if self.normalize:
                    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                num_captures += 1
                image = np.asarray(image)
                if(cap_shape == image.shape):
                    data.append(image)
                else:
                    count = count - 1
            count += 1
        if(len(data) == 0):
            data.append(first_image)
        return np.array(data)

