#!/usr/bin/env python3
"""Render the surface figures of the CAT website from CAT26 output.

  cutaway  Left hemisphere with cortical thickness on the pial surface. The
           back half is cut open along a sagittal T1 slice that shows the
           pial, central and white matter surfaces.
  folding  Gyrification, sulcal depth and cortical complexity (fractal
           dimension) on the central surface, where CAT measures them, in
           the same view and with the same colormap. Sulcal depth uses the
           colormap the other way round: deep sulci are dark, as in the
           shading of the surface.

Example (subject HR075):

  python3 tools/surface_figure.py --surf ~/Dropbox/T1/CAT26/surf \\
      --t1 ~/Dropbox/T1/HR075_MPRAGE.nii --name HR075_MPRAGE \\
      --cat ~/GitHub/cat12 --out /tmp/surface-figure

The CAT binaries (CAT_BlurSurfHK, CAT_DumpCurv, CAT_SulcusDepth and
CAT_FractalDimension) compute the measures exactly as cat_surf_parameters.m
does; intermediate files are cached per input in <out>/work. CAT26 and T1Prep
results use the same file names, so --surf can point to either. Rendering
uses VTK off screen. Requires numpy, scipy, nibabel, matplotlib, pillow and vtk.
"""
import argparse
import hashlib
import os
import platform
import subprocess
import sys

import numpy as np
import nibabel as nib
from nibabel.freesurfer.io import read_morph_data, write_morph_data
from scipy import ndimage
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

# surface contours on the T1 slice (pial and white as in the site's surface schematic)
CONTOUR = {'pial': '#11b2fc', 'central': '#ffb43c', 'white': '#3ddc2f'}
# camera: vector from the brain to the camera, focal point and zoom (perspective)
VIEW = dict(direction=(-0.9, 0.25, 0.35), focal=(-38, -12, 8), angle=22, zoom=1.5)
# sagittal window: lateral cap (x < CUT_X) removed posterior of y = CUT_Y
CUT_X, CUT_Y = -40, -10
FACE_BOX = (-108, 82, -60, 82)        # y0, y1, z0, z1 of the slice texture (mm)
FACE_RES = 0.1                        # mm per texture pixel


# ------------------------------------------------------------------ CAT ---
def cat_bin(cat):
    sysname, machine = platform.system(), platform.machine()
    sub = {'Darwin': 'CAT.maca64' if machine == 'arm64' else 'CAT.maci64',
           'Linux': 'CAT.glnx86', 'Windows': 'CAT.w32'}[sysname]
    return os.path.join(cat, sub)


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True, stdout=subprocess.DEVNULL)


def stale(out, *deps):
    """True if out is missing or older than one of its inputs."""
    return not os.path.exists(out) or any(os.path.getmtime(d) > os.path.getmtime(out) for d in deps)


def prepare(a):
    """Smoothed thickness, curvature, cortex mask and (optionally) folding
    measures on the native mesh of the left hemisphere. They are cached in a
    work folder per input (surf folder and T1), so CAT and T1Prep results of
    the same subject never mix, and recomputed when an input is newer."""
    key = hashlib.sha1(f'{os.path.abspath(a.surf)}|{os.path.abspath(a.t1)}'.encode()).hexdigest()[:8]
    w = os.path.join(a.out, 'work', f'{a.name}-{key}')
    os.makedirs(w, exist_ok=True)
    b = cat_bin(a.cat)
    tpl = os.path.join(a.cat, 'templates_surfaces')
    f = lambda kind, ext='': os.path.join(a.surf, f'lh.{kind}.{a.name}{ext}')
    o = lambda kind: os.path.join(w, f'lh.{kind}.{a.name}')
    central, sphere = f('central', '.gii'), f('sphere', '.gii')

    if stale(o('mask'), f('sphere.reg', '.gii')):
        # cortex mask of the template, mapped with the registered sphere (nearest vertex)
        ts, _ = load_gii(os.path.join(tpl, 'lh.sphere.freesurfer.gii'))
        sr, _ = load_gii(f('sphere.reg', '.gii'))
        unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        _, idx = cKDTree(unit(ts)).query(unit(sr))
        write_morph_data(o('mask'), read_morph_data(os.path.join(tpl, 'lh.mask'))[idx].astype('>f4'))
    if stale(o('curv'), central):
        run(os.path.join(b, 'CAT_DumpCurv'), central, o('curv'), 0, 0, 0)
    if stale(o('s5thickness'), central, f('thickness')):
        run(os.path.join(b, 'CAT_BlurSurfHK'), central, o('s5thickness'), 5, f('thickness'))
    if 'folding' in a.figures:
        if stale(o('gyrification'), central):
            run(os.path.join(b, 'CAT_DumpCurv'), central, o('gyrification'), 0, 0, 1)
        if stale(o('depth'), central, sphere):
            run(os.path.join(b, 'CAT_SulcusDepth'), central, sphere, o('depth'))
        if stale(o('fractaldimension'), central, sphere):
            run(os.path.join(b, 'CAT_FractalDimension'), '-sphere', sphere, '-nosmooth', central, sphere,
                o('fractaldimension'))
        for m, (fwhm, _, _) in FOLDING.items():
            if stale(o(f's{fwhm}{m}'), o(m), o('mask')):
                run(os.path.join(b, 'CAT_BlurSurfHK'), central, o(f's{fwhm}{m}'), fwhm, o(m), o('mask'))
    return o


# --------------------------------------------------------------- colors ---
def _oklab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = lin @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                          [0.2119034982, 0.6806995451, 0.1073969566],
                          [0.0883024619, 0.2817188376, 0.6299787005]]).T
    return np.cbrt(lms) @ np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                                    [1.9779984951, -2.4285922050, 0.4505937099],
                                    [0.0259040371, 0.7827717662, -0.8086757660]]).T


def _srgb(lab):
    lms = (lab @ np.array([[1.0, 0.3963377774, 0.2158037573],
                           [1.0, -0.1055613458, -0.0638541728],
                           [1.0, -0.0894841775, -1.2914855480]]).T) ** 3
    lin = np.clip(lms @ np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                                  [-1.2684380046, 2.6097574011, -0.3413193965],
                                  [-0.0041960863, -0.7034186147, 1.7076147010]]).T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def ramp(hexes, n=256):
    """Colormap from anchor colors, interpolated in OKLab (monotone lightness)."""
    anchors = _oklab(np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in hexes]))
    t = np.linspace(0, 1, len(hexes))
    tt = np.linspace(0, 1, n)
    return _srgb(np.stack([np.interp(tt, t, anchors[:, k]) for k in range(3)], 1))


HOT = ramp(['#2b0b3f', '#6a1a66', '#b02a5b', '#e8553f', '#f8a35d', '#fde3a7'])


def underlay(curv, base=0.80, depth=0.22):
    """Gray shading from CAT mean curvature: gyri (curv > 0) light, sulci darker."""
    return base + depth * 0.5 * (np.tanh(curv / 35.0) - 1) * 0.9


def colorize(values, vmin, vmax, curv, shade=0.30):
    """Values -> RGB; curvature darkens sulci a little so the folding stays
    readable under the colors; NaN (medial wall) shows the gray underlay."""
    t = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    rgb = HOT[np.nan_to_num(t * (len(HOT) - 1), nan=0).astype(int)]
    g = underlay(curv)
    rgb = rgb * ((1 - shade) + shade * (g - g.min()) / (np.ptp(g) + 1e-9))[:, None]
    nan = ~np.isfinite(values)
    rgb[nan] = g[nan, None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ VTK ---
def load_gii(path):
    d = {nib.nifti1.intent_codes.label[a.intent]: a.data for a in nib.load(path).darrays}
    return d['pointset'].astype(np.float64), d['triangle'].astype(np.int64)


def polydata(v, f, rgb=None, scalars=None):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(v), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(f) + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(f.ravel()), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    if rgb is not None:
        pd.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(rgb), deep=True,
                                                  array_type=vtk.VTK_UNSIGNED_CHAR))
    if scalars is not None:
        arr = numpy_to_vtk(scalars.astype(np.float64), deep=True)
        arr.SetName('s')
        pd.GetPointData().AddArray(arr)
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(pd)
    nf.SplittingOff()
    nf.ConsistencyOn()
    nf.AutoOrientNormalsOn()
    nf.Update()
    return nf.GetOutput()


def surface_actor(pd):
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(pd)
    m.SetColorModeToDirectScalars()
    act = vtk.vtkActor()
    act.SetMapper(m)
    p = act.GetProperty()
    p.SetInterpolationToPhong()
    p.SetAmbient(0.18)
    p.SetDiffuse(0.82)
    p.SetSpecular(0.10)
    p.SetSpecularPower(18)
    return act


def plane(origin, normal):
    p = vtk.vtkPlane()
    p.SetOrigin(*origin)
    p.SetNormal(*normal)
    return p


def render(actors, size, ss=2):
    """Light kit, ambient occlusion, supersampling; RGBA image trimmed to the content."""
    ren = vtk.vtkRenderer()
    for act in actors:
        ren.AddActor(act)
    ren.SetBackground(1, 1, 1)
    ren.SetBackgroundAlpha(0)
    ren.SetUseDepthPeeling(1)
    lk = vtk.vtkLightKit()
    lk.SetKeyLightIntensity(0.95)
    lk.SetKeyToFillRatio(2.6)
    lk.SetKeyToHeadRatio(3.2)
    lk.SetKeyLightElevation(45)
    lk.SetKeyLightAzimuth(15)
    lk.AddLightsToRenderer(ren)
    ren.UseSSAOOn()
    ren.SetSSAORadius(5.0)
    ren.SetSSAOBias(0.5)                # avoids banding (self-occlusion) on the smooth surface
    ren.SetSSAOKernelSize(256)
    ren.SSAOBlurOn()
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetAlphaBitPlanes(1)
    win.SetMultiSamples(0)
    win.AddRenderer(ren)
    win.SetSize(size[0] * ss, size[1] * ss)

    c = np.array(VIEW['focal'], float)
    d = np.array(VIEW['direction'], float)
    d /= np.linalg.norm(d)
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(*c)
    cam.SetPosition(*(c + 600 * d))
    cam.SetViewUp(0, 0, 1)
    cam.OrthogonalizeViewUp()
    cam.SetViewAngle(VIEW['angle'])
    ren.ResetCamera()
    cam.Dolly(VIEW['zoom'])
    cam.SetFocalPoint(*c)
    cam.SetPosition(*(c + cam.GetDistance() * d))
    ren.ResetCameraClippingRange()
    win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToRGBA()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, 4)[::-1]
    win.Finalize()
    im = Image.fromarray(np.ascontiguousarray(arr), 'RGBA').resize(size, Image.LANCZOS)
    x0, y0, x1, y1 = im.getchannel('A').getbbox()
    if x0 == 0 or y0 == 0 or x1 == size[0] or y1 == size[1]:
        print('warning: the brain touches the image border, lower VIEW["zoom"]')
    return im.crop((max(x0 - 8, 0), max(y0 - 8, 0), min(x1 + 8, size[0]), min(y1 + 8, size[1])))


# ----------------------------------------------------------- T1 slice ---
def cutter(v, f, x, scalars=None):
    c = vtk.vtkCutter()
    c.SetCutFunction(plane((x, 0, 0), (1, 0, 0)))
    c.SetInputData(polydata(v, f, scalars=scalars))
    c.Update()
    return c


def contour_segments(v, f, x, mask):
    """Cut of a surface with the sagittal plane as (y, z) segments, without the medial wall."""
    o = cutter(v, f, x, mask).GetOutput()
    pts = vtk_to_numpy(o.GetPoints().GetData())[:, 1:]
    m = vtk_to_numpy(o.GetPointData().GetArray('s'))
    ln = vtk_to_numpy(o.GetLines().GetConnectivityArray()).reshape(-1, 2)
    return pts[ln][m[ln].mean(1) > 0.5]


def contour_loops(v, f, x):
    c = cutter(v, f, x)
    st = vtk.vtkStripper()
    st.SetInputConnection(c.GetOutputPort())
    st.JoinContiguousSegmentsOn()
    st.Update()
    o = st.GetOutput()
    pts = vtk_to_numpy(o.GetPoints().GetData())[:, 1:]
    lines = o.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        yield pts[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]]


def face_texture(a, o, path, gamma=1.15, lw=4.0):
    """Sagittal T1 slice at CUT_X with the three surface contours (RGBA);
    transparent outside the pial surface and anterior of CUT_Y."""
    y0, y1, z0, z1 = FACE_BOX
    img = nib.load(a.t1)
    yy, zz = np.meshgrid(np.arange(y0, y1, FACE_RES), np.arange(z0, z1, FACE_RES))
    P = np.vstack([np.full(yy.size, CUT_X), yy.ravel(), zz.ravel(), np.ones(yy.size)])
    ijk = np.linalg.inv(img.affine) @ P
    t1 = ndimage.map_coordinates(np.asarray(img.dataobj, np.float32), ijk[:3], order=3).reshape(yy.shape)
    lo, hi = np.percentile(t1[t1 > 0], [1, 99.7])
    g = np.clip((t1 - lo) / (hi - lo), 0, 1) ** gamma
    H, W = g.shape

    surf = {k: load_gii(os.path.join(a.surf, f'lh.{k}.{a.name}.gii')) for k in CONTOUR}
    inside = np.zeros((H, W), bool)                      # even-odd fill of the pial cut
    for lp in contour_loops(*surf['pial'], CUT_X):
        m = Image.new('1', (W, H), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in np.c_[(lp[:, 0] - y0) / FACE_RES, (lp[:, 1] - z0) / FACE_RES]],
                                  fill=1)
        inside ^= np.asarray(m)
    # solid face: close cracks, fill enclosed holes, drop tiny islands, keep the window
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(inside, iterations=3))
    lab, _ = ndimage.label(inside)
    inside &= np.bincount(lab.ravel())[lab] >= 40 / FACE_RES ** 2
    inside &= yy < CUT_Y
    near = ndimage.binary_dilation(inside, iterations=4)   # keeps the contour lines on the border

    mask = read_morph_data(o('mask'))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.dstack([g, g, g, inside.astype(float)]), origin='lower', extent=FACE_BOX,
              interpolation='bilinear')
    for k in ('white', 'central', 'pial'):
        s = contour_segments(*surf[k], CUT_X, mask)
        mid = s.mean(1)
        r = np.clip(((mid[:, 1] - z0) / FACE_RES).astype(int), 0, H - 1)
        c = np.clip(((mid[:, 0] - y0) / FACE_RES).astype(int), 0, W - 1)
        ax.add_collection(LineCollection(s[near[r, c]], colors=CONTOUR[k], linewidths=lw,
                                         capstyle='round', joinstyle='round'))
    ax.set_xlim(y0, y1)
    ax.set_ylim(z0, z1)
    ax.axis('off')
    fig.savefig(path, dpi=100, transparent=True)
    plt.close(fig)


def face_actor(png):
    y0, y1, z0, z1 = FACE_BOX
    ps = vtk.vtkPlaneSource()
    x = CUT_X - 0.15                   # slightly lateral of the cut, towards the camera
    ps.SetOrigin(x, y0, z0)
    ps.SetPoint1(x, y1, z0)
    ps.SetPoint2(x, y0, z1)
    rd = vtk.vtkPNGReader()
    rd.SetFileName(png)
    tex = vtk.vtkTexture()
    tex.SetInputConnection(rd.GetOutputPort())
    tex.InterpolateOn()
    tex.MipmapOn()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(ps.GetOutputPort())
    act = vtk.vtkActor()
    act.SetMapper(m)
    act.SetTexture(tex)
    act.GetProperty().LightingOff()
    return act


# -------------------------------------------------------------- figures ---
def cortex_actor(a, o, kind, values, vmin, vmax, clip=None):
    """Surface (pial or central) coloured by values; vmin > vmax reverses the colormap."""
    v, f = load_gii(os.path.join(a.surf, f'lh.{kind}.{a.name}.gii'))
    if len(values) != len(v):
        sys.exit(f'lh.{kind}.{a.name}.gii has {len(v)} vertices, but the values have {len(values)}')
    values = values.copy()
    values[read_morph_data(o('mask')) < 0.5] = np.nan
    pd = polydata(v, f, colorize(values, vmin, vmax, read_morph_data(o('curv'))))
    if clip is not None:
        cl = vtk.vtkClipPolyData()
        cl.SetClipFunction(clip)
        cl.SetInputData(pd)
        cl.Update()
        pd = cl.GetOutput()
    return surface_actor(pd)


def save(im, path, webp_width):
    im.save(path + '.png')
    w = webp_width
    im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(
        path + '.webp', 'WEBP', quality=82, method=6, alpha_quality=90)
    print('wrote', path + '.png/.webp')


def cutaway(a, o):
    png = o('face') + '.png'               # recreated each time: it also depends on the cut settings
    face_texture(a, o, png)
    clip = vtk.vtkImplicitBoolean()                      # remove x < CUT_X and y < CUT_Y
    clip.SetOperationTypeToIntersection()
    clip.AddFunction(plane((CUT_X, 0, 0), (1, 0, 0)))
    clip.AddFunction(plane((0, CUT_Y, 0), (0, 1, 0)))
    brain = cortex_actor(a, o, 'pial', read_morph_data(o('s5thickness')), *THICKNESS_RANGE, clip=clip)
    save(render([brain, face_actor(png)], (2600, 1950)), os.path.join(a.out, 'surface-cutaway'), 1400)


def folding(a, o):
    mask = read_morph_data(o('mask')) > 0.5
    for m, (fwhm, direction, upper) in FOLDING.items():
        x = read_morph_data(o(f's{fwhm}{m}'))
        lo, hi = np.nanpercentile(x[mask], [2, upper])
        print(f'{m}: colormap range {lo:.3g} .. {hi:.3g}' + (' (reversed)' if direction < 0 else ''))
        rng = (lo, hi) if direction > 0 else (hi, lo)
        save(render([cortex_actor(a, o, 'central', x, *rng)], (1300, 975)), os.path.join(a.out, f'surface-{m}'), 480)


THICKNESS_RANGE = (1.8, 3.6)          # mm, as in the figure key
# folding measure: smoothing (FWHM in mm), colormap direction (-1: high values dark) and
# upper percentile of the colormap range (sulcal depth is skewed towards shallow values)
FOLDING = {'gyrification': (20, 1, 98), 'depth': (15, -1, 90), 'fractaldimension': (20, 1, 98)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--surf', required=True, help='CAT surf folder of the subject')
    ap.add_argument('--t1', required=True, help='T1 image that CAT processed (same space as the surfaces)')
    ap.add_argument('--name', required=True, help='subject name in the CAT file names, e.g. HR075_MPRAGE')
    ap.add_argument('--cat', required=True, help='CAT toolbox folder (binaries and templates_surfaces)')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--figures', nargs='+', default=['cutaway', 'folding'], choices=['cutaway', 'folding'])
    a = ap.parse_args()
    for k in ('surf', 't1', 'cat', 'out'):
        setattr(a, k, os.path.expanduser(getattr(a, k)))
    o = prepare(a)
    if 'cutaway' in a.figures:
        cutaway(a, o)
    if 'folding' in a.figures:
        folding(a, o)


if __name__ == '__main__':
    main()
