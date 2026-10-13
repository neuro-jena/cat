#!/usr/bin/env python3
"""Render the two panels of the ROI figure of the CAT website and print its values.

  volume   A coronal T1 slice with the gray matter regions of the
           Neuromorphometrics atlas in native space, as CAT writes it.
  surface  The left pial surface with the Desikan-Killiany atlas, mapped from
           the template through the subject's spherical registration.

The native atlas needs a CAT run with "Atlas label maps > native" switched on,
for example (MATLAB, SPM and CAT on the path; writes mri/, label/, report/
next to the image):

  global cat; cat_get_defaults; job = cat;
  job.data = {'/path/to/HR075_MPRAGE.nii'}; job.nproc = 0;
  job.output.surface = 0; job.output.atlas.native = 1;
  cat_run(job);

Example (subject HR075):

  python3 tools/roi_figure.py --vol /tmp/cat26 --surf ~/Dropbox/T1/CAT26/surf \\
      --name HR075_MPRAGE --cat ~/spm/spm12/toolbox/CAT --out /tmp/roi-figure

--vol is the folder of that run (the T1 image with mri/ and label/ below it),
--surf the surf folder of a run with surfaces (label/catROIs_<name>.xml next to
it holds the regional thickness).  Region colours: gray matter regions only,
left and right alike, and structures that touch get colours that differ as
much as possible.  Requires numpy, scipy, nibabel, pillow and vtk.
"""
import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import nibabel as nib
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from nibabel.freesurfer.io import read_morph_data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surface_figure as sf     # noqa: E402  renderer, shading and CAT helpers

#: (atlas region, label in the table) of the two tables, left hemisphere
VOLUME_ROWS = [('Left Hippocampus', 'Hippocampus'), ('Left Putamen', 'Putamen'),
               ('Left STG superior temporal gyrus', 'Superior temporal')]
SURFACE_ROWS = [('precentral', 'Precentral'), ('postcentral', 'Postcentral'),
                ('superiortemporal', 'Superior temporal')]
SURFACE_VIEW = dict(direction=(-1.0, 0.0, 0.12), focal=(-35, -18, 15), angle=22, zoom=1.25)


# ------------------------------------------------------------ colours ---
def oklch(L, C, h):
    a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
    return sf._srgb(np.array([[L, a, b]]))[0]


def region_colours(atlas_csv, lab):
    """Colour per gray matter region of the atlas, same for both hemispheres;
    greedy colouring over the 3D adjacency in *lab* keeps neighbours apart."""
    names = {}
    with open(atlas_csv, encoding='utf-8') as fh:
        for row in csv.DictReader(fh, delimiter=';'):
            if row['Vgm'] == '1':
                names[int(row['ROIid'])] = row['ROIname']
    structure = {rid: n.replace('Left ', '').replace('Right ', '') for rid, n in names.items()}
    bases = sorted(set(structure.values()))
    neighbours = {b: set() for b in bases}
    codes = []
    for ax in range(3):
        a = np.moveaxis(lab, ax, 0)
        x, y = a[1:].ravel(), a[:-1].ravel()
        m = (x != y) & (x > 0) & (y > 0)
        codes.append(np.unique(x[m].astype(np.int64) * 1000 + y[m]))
    for code in np.unique(np.concatenate(codes)):
        p, q = divmod(int(code), 1000)
        if p in structure and q in structure and structure[p] != structure[q]:
            neighbours[structure[p]].add(structure[q])
            neighbours[structure[q]].add(structure[p])
    candidates = np.array([oklch(L, 0.13, h) for L in (0.74, 0.62) for h in range(0, 360, 20)])
    lab_c = sf._oklab(candidates)
    used = np.zeros(len(candidates))
    colour = {}
    for b in sorted(bases, key=lambda b: -len(neighbours[b])):
        taken = [colour[n] for n in neighbours[b] if n in colour]
        d = (np.min(np.linalg.norm(lab_c[:, None] - lab_c[taken][None], axis=2), axis=1)
             if taken else np.ones(len(candidates)))
        colour[b] = int(np.argmax(d - 0.01 * used))     # most distinct, then the least used
        used[colour[b]] += 1
    return {rid: (names[rid], candidates[colour[structure[rid]]]) for rid in names}


def hex_colour(rgb):
    return '#%02x%02x%02x' % tuple((np.clip(rgb, 0, 1) * 255).round().astype(int))


# ------------------------------------------------------------- volume ---
def ras(path):
    """Image data in RAS+ voxel order (x right, y anterior, z superior)."""
    return np.asanyarray(nib.as_closest_canonical(nib.load(path)).dataobj)


def coronal(volume, j):
    """Coronal plane at RAS row j: superior up, left hemisphere on the left."""
    return np.flipud(volume[:, j, :].T)


def volume_panel(a, path, alpha=0.6, up=4):
    lab = ras(os.path.join(a.vol, 'mri', f'neuromorphometrics_{a.name}.nii')).astype(int)
    t1 = ras(os.path.join(a.vol, f'{a.name}.nii')).astype(float)
    p0 = ras(os.path.join(a.vol, 'mri', f'p0{a.name}.nii')).astype(float)
    table = region_colours(os.path.join(a.cat, 'templates_MNI152NLin2009cAsym', 'neuromorphometrics.csv'), lab)
    j = a.slice if a.slice is not None else int(np.argmax(np.isin(lab, (47, 48, 59, 60)).sum(axis=(0, 2))))
    L, g, tissue = coronal(lab, j), coronal(t1, j), coronal(p0, j)
    lo, hi = np.percentile(g[tissue > 0.5], [1, 99.5])
    g = np.clip(ndimage.zoom(np.clip((g - lo) / (hi - lo), 0, 1), up, order=3), 0, 1)
    # smooth region borders: every region upsampled on its own, the strongest wins
    best, U = np.full(g.shape, 0.5), np.zeros(g.shape, int)
    for rid in np.unique(L):
        if rid in table:
            m = ndimage.zoom((L == rid).astype(float), up, order=1)
            take = m > best
            best[take], U[take] = m[take], rid
    rgb = np.repeat(g[..., None], 3, axis=2)
    for rid in np.unique(U[U > 0]):
        m = U == rid
        rgb[m] = (1 - alpha) * rgb[m] + alpha * table[rid][1]
    edge = np.zeros(U.shape, bool)                       # thin, darker borders between regions
    for ax in (0, 1):
        d = np.diff(U, axis=ax) != 0
        lead, trail = [slice(None)] * 2, [slice(None)] * 2
        lead[ax], trail[ax] = slice(1, None), slice(None, -1)
        edge[tuple(lead)] |= d
        edge[tuple(trail)] |= d
    rgb[edge & (U > 0)] *= 0.6
    brain = ndimage.zoom((tissue > 1.5).astype(float), up, order=1) > 0.5
    mask = ndimage.binary_fill_holes(ndimage.binary_opening(brain | (U > 0), iterations=2))
    im = Image.fromarray((np.clip(np.dstack([rgb, mask]), 0, 1) * 255).astype(np.uint8), 'RGBA')
    save(im.crop(im.getchannel('A').getbbox()), path, 560)
    print(f'volume panel: coronal slice {j} (RAS voxel row)')
    return table


# ------------------------------------------------------------ surface ---
def surface_panel(a, path):
    labels, ctab, names = nib.freesurfer.read_annot(
        os.path.join(a.cat, 'atlases_surfaces', 'lh.aparc_DK40.freesurfer.annot'))
    tpl, _ = sf.load_gii(os.path.join(a.cat, 'templates_surfaces', 'lh.sphere.freesurfer.gii'))
    reg, _ = sf.load_gii(os.path.join(a.surf, f'lh.sphere.reg.{a.name}.gii'))
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    _, idx = cKDTree(unit(tpl)).query(unit(reg))        # template label at each subject vertex
    labels, table = labels[idx], ctab[:, :3] / 255.0
    pial = os.path.join(a.surf, f'lh.pial.{a.name}.gii')
    curv = os.path.join(a.out, f'lh.curv_pial.{a.name}')
    if not os.path.exists(curv):
        sf.run(os.path.join(sf.cat_bin(a.cat), 'CAT_DumpCurv'), pial, curv, 0, 0, 0)
    grey = sf.underlay(read_morph_data(curv))
    shade = 0.68 + 0.32 * (grey - grey.min()) / (np.ptp(grey) + 1e-9)    # sulci a little darker
    rgb = np.repeat(grey[:, None], 3, axis=1)
    ok = (labels >= 0) & (labels < len(table))
    rgb[ok] = table[labels[ok]] * shade[ok, None]
    v, f = sf.load_gii(pial)
    sf.VIEW.update(SURFACE_VIEW)
    im = sf.render([sf.surface_actor(sf.polydata(v, f, (np.clip(rgb, 0, 1) * 255).astype(np.uint8)))], (2400, 1700))
    save(im, path, 640)
    return {n.decode(): table[i] for i, n in enumerate(names)}


def save(im, path, width):
    im.save(path + '.png')
    im.resize((width, round(im.height * width / im.width)), Image.LANCZOS).save(
        path + '.webp', 'WEBP', quality=88, method=6, alpha_quality=90)
    print('wrote', path + '.png/.webp')


# ------------------------------------------------------------- values ---
def roi_values(xml, atlas, measure):
    node = ET.parse(xml).getroot().find(atlas)
    names = [i.text for i in node.find('names')]
    values = [float(x) for x in node.find(f'data/{measure}').text.strip('[]').split(';')]
    return dict(zip(names, values))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vol', required=True, help='folder of the CAT run with the native atlas (T1, mri/, label/)')
    ap.add_argument('--surf', required=True, help='CAT surf folder of the subject')
    ap.add_argument('--name', required=True, help='subject name in the CAT file names, e.g. HR075_MPRAGE')
    ap.add_argument('--cat', required=True, help='CAT toolbox folder (binaries, atlases and templates)')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--slice', type=int, help='coronal slice (RAS voxel row); default: most hippocampus and thalamus')
    a = ap.parse_args()
    for k in ('vol', 'surf', 'cat', 'out'):
        setattr(a, k, os.path.expanduser(getattr(a, k)))
    os.makedirs(a.out, exist_ok=True)

    table = volume_panel(a, os.path.join(a.out, 'roi-volume'))
    gm = roi_values(os.path.join(a.vol, 'label', f'catROI_{a.name}.xml'), 'neuromorphometrics', 'Vgm')
    ids = {name: rid for rid, (name, _c) in table.items()}
    print('\nGray matter (ml):')
    for region, label in VOLUME_ROWS:
        print(f'  {hex_colour(table[ids[region]][1])}  {label:20s} {gm[region]:.2f}')

    colours = surface_panel(a, os.path.join(a.out, 'roi-surface'))
    th = roi_values(os.path.join(os.path.dirname(a.surf.rstrip('/')), 'label', f'catROIs_{a.name}.xml'),
                    'aparc_DK40', 'thickness')
    print('\nThickness (mm):')
    for region, label in SURFACE_ROWS:
        print(f'  {hex_colour(colours[region])}  {label:20s} {th["l" + region]:.2f}')


if __name__ == '__main__':
    main()
