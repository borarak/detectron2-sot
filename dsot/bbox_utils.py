from collections import namedtuple
import torch
import numpy as np

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def get_bbox(image, shape, path=None, wc_call=None, z_row=None):
    try:
        imh, imw = image.shape[:2]
    except:
        raise ValueError(
            f"Failed when reading {path} and in call {wc_call} and z_row is {z_row}"
        )
    if len(shape) == 4:
        w, h = shape[2] - shape[0], shape[3] - shape[1]
    else:
        w, h = shape
    context_amount = 0.5
    exemplar_size = 127
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    w = w * scale_z
    h = h * scale_z
    cx, cy = imw // 2, imh // 2
    bbox = center2corner(Center(cx, cy, w, h))
    return bbox


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
  Args:
      center: Center or np.array (4 * N)
  Return:
      center or np.array (4 * N)
  """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def bbox2tensor(bbox):
    return torch.tensor([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
