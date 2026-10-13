#!/usr/bin/env python3
"""Render the five tiles of the VBM figure of the CAT website.

A sagittal slice of the T1 image, CAT's partial-volume label map (p0) and the
gray matter, white matter and CSF fractions derived from it, in one set of
tissue colours: CSF blue, gray matter red-orange, white matter cream.  The
label map blends them where a voxel holds two tissues.

Example (subject HR075):

  python3 tools/vbm_figure.py --t1 ~/Dropbox/T1/HR075_MPRAGE.nii \\
      --p0 ~/Dropbox/T1/CAT26/mri/p0HR075_MPRAGE.nii --out /tmp/vbm-figure

Writes vbm-t1, vbm-label (520 px wide) and vbm-gm, vbm-wm, vbm-csf (360 px)
as PNG and WebP.  Requires numpy, nibabel and pillow.
"""
import argparse
import os

import numpy as np
import nibabel as nib
from PIL import Image

CSF = np.array([0x4f, 0x9b, 0xe6]) / 255
GM = np.array([0xe8, 0x55, 0x3f]) / 255
WM = np.array([0xfd, 0xe3, 0xa7]) / 255
WIDTHS = {'vbm-t1': 520, 'vbm-label': 520, 'vbm-gm': 360, 'vbm-wm': 360, 'vbm-csf': 360}


def ras(path):
    """Image data in RAS+ voxel order (x right, y anterior, z superior) and the voxel size."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj).astype(float), img.header.get_zooms()[:3]


def sagittal(vol, i):
    """Sagittal plane at RAS column i: anterior on the left, superior up."""
    return np.flipud(np.fliplr(vol[i, :, :].T))


def midline(p0):
    """x index of the midline: the slice with the least tissue between the hemispheres."""
    tissue = (p0 > 1.5).sum(axis=(1, 2)).astype(float)
    centre = int(round(np.average(np.arange(len(tissue)), weights=tissue)))
    return min(range(centre - 15, centre + 16), key=lambda i: tissue[i])


def tissues(p0):
    """CAT's partial-volume label (1 CSF, 2 GM, 3 WM) -> tissue fractions."""
    csf = np.clip(1 - np.abs(p0 - 1), 0, 1)
    gm = np.clip(1 - np.abs(p0 - 2), 0, 1)
    wm = np.clip(1 - np.abs(p0 - 3), 0, 1) * (p0 > 2)
    return csf, gm, wm


def label_colours(p0):
    """Black -> CSF -> GM -> WM along the label, so partial volumes blend."""
    cols = np.stack([np.zeros(3), CSF, GM, WM])
    return np.stack([np.interp(np.clip(p0, 0, 3), [0, 1, 2, 3], cols[:, k]) for k in range(3)], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--t1', required=True, help='T1 image that CAT processed')
    ap.add_argument('--p0', required=True, help="CAT's label map p0<name>.nii in native space")
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--offset', type=float, default=5.0, help='slice distance left of the midline in mm (default 5)')
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    t1, zooms = ras(os.path.expanduser(a.t1))
    p0, _ = ras(os.path.expanduser(a.p0))
    i = midline(p0) - int(round(a.offset / zooms[0]))           # left hemisphere: smaller x
    label = sagittal(p0, i)
    ys, xs = np.where(label > 0.5)
    margin = int(round(8 / zooms[1]))
    y0, y1, x0, x1 = max(ys.min() - margin, 0), ys.max() + margin, max(xs.min() - margin, 0), xs.max() + margin
    g = sagittal(t1, i)
    lo, hi = np.percentile(g[g > np.percentile(g, 30)], [0.5, 99.7])
    grey = np.clip((g - lo) / (hi - lo), 0, 1)
    csf, gm, wm = tissues(label)
    tiles = {'vbm-t1': np.repeat(grey[..., None], 3, -1), 'vbm-label': label_colours(label),
             'vbm-gm': gm[..., None] * GM, 'vbm-wm': wm[..., None] * WM, 'vbm-csf': csf[..., None] * CSF}
    for name, rgb in tiles.items():
        im = Image.fromarray((np.clip(rgb[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8))
        im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)        # smooth, then scale to the web size
        im.save(os.path.join(out, name + '.png'))
        w = WIDTHS[name]
        im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(
            os.path.join(out, name + '.webp'), 'WEBP', quality=88, method=6)
    print(f'sagittal slice x = {i} (RAS voxel), {a.offset:g} mm left of the midline; wrote {out}')


if __name__ == '__main__':
    main()
