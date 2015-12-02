#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from matplotlib.backend_bases import KeyEvent, MouseEvent
from math import ceil, floor, exp
from stl import mesh
from os import rename
from os.path import basename

MARGIN = 0.02
INITIAL_INFLATION = 1.5
RESOLUTION = 10.0
OUTPUT = 'out.stl'

def chunks(l, n, m):
    i = 0
    for _ in range(0, m):
        yield l[i:i+n]
        i += n

def real_to_ref(min, max, x, margin):
    return margin + (1.0 - 2*margin) * (x - min) / (max - min)

def real_to_ref_a(box, x, margin):
    return np.array((real_to_ref(box[0], box[1], x[0], margin),
                     real_to_ref(box[2], box[3], x[1], margin)))

def real_to_ref_db(box, x, y, margin):
    return (real_to_ref(box[0], box[1], x, margin),
            real_to_ref(box[2], box[3], y, margin))

def ref_to_real(min, max, x, margin):
    result = (x - margin) / (1.0 - 2*margin) * (max - min) + min
    return result

def ref_to_real_a(box, x, margin):
    return np.array((ref_to_real(box[0], box[1], x[0], margin),
                     ref_to_real(box[2], box[3], x[1], margin)))

def ref_to_real_db(box, x, y, margin):
    return (ref_to_real(box[0], box[1], x, margin),
            ref_to_real(box[2], box[3], y, margin))

def compute_box(center, corner, box, margin, scale=1.0):
    center = ref_to_real_a(box, center, margin)
    corner = ref_to_real_a(box, corner, margin)

    up = (center - corner) * scale
    corner = center - up
    right = np.array((up[1], -up[0]))

    se = corner - right
    sw = real_to_ref_a(box, se + 2*right, margin)
    ne = real_to_ref_a(box, se + 2*up, margin)
    nw = real_to_ref_a(box, se + 2*up + 2*right, margin)
    se = real_to_ref_a(box, se, margin)

    return se, ne, nw, sw

def dem_parse(spec, s, init):
    res = {}

    for start, end, attr, kind, optional in spec:
        piece = s[(init+start):(init+end)]
        if kind is str:
            value = piece.decode().strip()
        elif kind is int or type(kind) is dict:
            try:
                value = int(piece)
                if type(kind) is dict:
                    value = kind[value]
            except (ValueError, IndexError) as e:
                if optional:
                    value = None
                else:
                    raise e
        elif kind is float:
            try:
                value = float(piece)
            except:
                if optional:
                    value = None
                else:
                    raise e
        elif kind == 'list_float':
            value = [float(q) for q in piece.decode().split()]
        elif kind == 'list_int':
            value = [int(q) for q in piece.decode().split()]
        res[attr] = value

    return res

class Progress:
    def __init__(self):
        self.n = 0

    def __call__(self, message, end=False):
        sys.stdout.write('\r' + ' '*self.n)
        sys.stdout.write('\r' + message)
        if end:
            sys.stdout.write('\n')
        sys.stdout.flush()
        self.n = len(message)

class DEMFile:
    def __init__(self, fn, submap=None, out=False):
        if isinstance(fn, str):
            self._from_dem(fn, out)
        elif isinstance(fn, h5py.File):
            self._from_hdf5(fn, submap)

    def _from_dem(self, fn, out):
        with open(fn, 'rb') as f:
            s = f.read()
        self._load_record_a(s[:1024], out)
        self._check()
        self.data = []
        self.file_name = basename(fn).split('.')[0]

        init = 1024
        p = Progress()
        for i in range(0, self.size):
            p('{}: reading profile {}/{}'.format(fn, i+1, self.size))
            init = self._load_data(s, init)

        self.data = np.rot90(np.array(self.data))

        p('{}: finished ({}×{})'.format(fn, *self.data.shape), end=True)

    def _from_hdf5(self, f, submap):
        candidates = [g for g in f['maps'] if g.startswith(submap)]
        assert len(candidates) == 1
        grp = f['maps'][candidates[0]]

        self.data = grp['data'][()]
        for v in ['east', 'south', 'north', 'west']:
            setattr(self, v, grp[v][()])

    def push_to_hdf5(self, f):
        maps = f.require_group('maps')
        grp = maps.require_group(self.file_name)

        for v in ['data', 'east', 'south', 'north', 'west']:
            grp[v] = getattr(self, v)

    def plot(self, fig, box, vmin, vmax):
        east = real_to_ref(box[0], box[1], self.east, MARGIN)
        west = real_to_ref(box[0], box[1], self.west, MARGIN)
        south = real_to_ref(box[2], box[3], self.south, MARGIN)
        north = real_to_ref(box[2], box[3], self.north, MARGIN)

        ax = fig.add_axes((east, south, west-east, north-south))
        ax.imshow(self.data, cmap='terrain', aspect='auto',
                  extent=(self.east, self.west,
                          self.south, self.north),
                  vmin=vmin, vmax=vmax)
        ax.axis('off')

    def elevation(self, xpts, ypts):
        i = (1 - real_to_ref(self.south, self.north, ypts, 0.0)) * (self.data.shape[0] - 1)
        j = real_to_ref(self.east, self.west, xpts, 0.0) * (self.data.shape[1] - 1)

        ir = np.floor(i).astype(int)
        jr = np.floor(j).astype(int)

        ir[np.nonzero(ir == (self.data.shape[0] - 1))] -= 1
        jr[np.nonzero(jr == (self.data.shape[1] - 1))] -= 1

        return (self.data[ir, jr] * (1 - (i - ir)) * (1 - (j - jr)) +
                self.data[ir+1, jr] * (i - ir) * (1 - (j - jr)) +
                self.data[ir, jr+1] * (1 - (i - ir)) * (j - jr) +
                self.data[ir+1, jr+1] * (i - ir) * (j - jr))

    def _load_record_a(self, s, out):
        spec = [
            (0, 40, 'file_name', str, False),
            (144, 150, 'level', {1: 'DEM-1', 2: 'DEM-2', 3: 'DEM-3', 4: 'DEM-4'}, False),
            (150, 156, 'pattern', {1: 'regular', 2: 'random'}, False),
            (156, 162, 'reference_system', {0: 'geographic', 1: 'UTM', 2: 'state plane'}, False),
            (162, 168, 'zone', int, False),
            (168, 528, 'projection_parameters', 'list_float', False),
            (528, 534, 'ground_unit', {0: 'radians', 1: 'feet', 2: 'meters', 3: 'arcseconds'}, False),
            (528, 534, 'elevation_unit', {0: 'radians', 1: 'feet', 2: 'meters', 3: 'arcseconds'}, False),
            (541, 546, 'sides', int, False),
            (546, 738, 'corners', 'list_float', False),
            (738, 786, 'elevations', 'list_float', False),
            (786, 810, 'reference_angle', float, False),
            (810, 816, 'accuracy', {0: 'unknown', 1: 'known'}, False),
            (816, 852, 'resolution', 'list_float', False),
            (852, 864, 'size', 'list_int', False),
        ]
        params = dem_parse(spec, s, 0)
        self.__dict__.update(params)

        if out:
            for _, _, attr, _, _ in spec:
                name = ' '.join(attr.split('_'))
                name = name[0].upper() + name[1:]
                print('{}:'.format(name), getattr(self, attr))

    def _load_data(self, s, init):
        spec = [
            (0, 12, 'pos', 'list_int', False),
            (12, 24, 'points', 'list_int', False),
            (24, 72, 'initial', 'list_float', False),
            (72, 96, 'local', float, False),
            (96, 144, 'limits', 'list_float', False),
        ]
        params = dem_parse(spec, s, init)

        assert len(params['pos']) == 2
        assert params['pos'][0] == 1
        assert 1 <= params['pos'][0] <= self.size
        assert len(params['points']) == 2
        assert params['points'][1] == 1
        assert len(params['limits']) == 2

        size = params['points'][0]
        pos = params['pos'][1]

        data = []
        for v in chunks(s[init+144:init+1024].decode(), 6, 146):
            try:
                data.append(int(v))
            except ValueError:
                break
        while len(data) < size:
            init += 1024
            for v in chunks(s[init:init+1024].decode(), 6, 170):
                try:
                    data.append(int(v))
                except ValueError:
                    break

        data = [(0 if v == -32767 else v) * self.resolution[2] + params['local']
                for v in data]
        assert(len(data) == size)

        if data:
            npts_before = (params['initial'][1] - self.south) / self.resolution[1]
            npts_after = (self.north - params['initial'][1]) / self.resolution[1] - size + 1
            data = [data[0]]*int(npts_before) + data + [data[-1]]*int(npts_after)
        else:
            npts = (self.north - self.south) / self.resolution[1] + 1
            data = [0.0]*int(npts)

        self.data.append(data)
        return init + 1024

    def _check(self):
        assert self.pattern == 'regular'
        assert self.reference_system == 'UTM'
        assert all(v == 0.0 for v in self.projection_parameters)
        assert self.ground_unit == 'meters'
        assert self.elevation_unit == 'meters'
        assert self.sides == 4
        assert len(self.corners) == 8
        assert self.corners[0] == self.corners[2] < self.corners[4] == self.corners[6]
        assert self.corners[1] == self.corners[7] < self.corners[3] == self.corners[5]
        assert len(self.elevations) == 2
        assert self.elevations[0] <= self.elevations[1]
        assert self.reference_angle == 0.0
        assert len(self.size) == 2
        assert self.size[0] == 1

        self.size = self.size[1]
        self.east = self.corners[0]
        self.west = self.corners[4]
        self.south = self.corners[1]
        self.north = self.corners[3]


class ClickHandler:

    def __init__(self, files, ax, box):
        self.files = files
        self.center = None
        self.corner = None
        self.inflation = INITIAL_INFLATION
        self.ax = ax
        self.box = box
        self.output = []

        parts = OUTPUT.split('.')
        self.pattern = '.'.join(parts[:-1]) + '{:0{}}.' + parts[-1]

    def _get_next_filename(self):
        if not self.output:
            return OUTPUT

        next_num = len(self.output) + 1
        ndigits = len(str(next_num))
        if ndigits > len(str(len(self.output))) or len(self.output) == 1:
            new_output = [self.pattern.format(i+1, ndigits)
                          for i in range(len(self.output))]
            for old, new in zip(self.output, new_output):
                rename(old, new)
            self.output = new_output

        return self.pattern.format(next_num, ndigits)

    def _handle_mouse(self, event):
        if event.button == 1:
            self.corner = np.array((event.xdata, event.ydata))
        elif event.button == 3:
            self.center = np.array((event.xdata, event.ydata))
        elif event.button in ['up', 'down']:
            self.inflation += event.step / 50
            self.inflation = max(1.0, self.inflation)

    def _handle_key(self, event):
        if event.key == 'e':
            self._export_sdl()
        elif event.key == 'y':
            self._export_sdl(print_res=True)
        elif event.key == 'q':
            if self.output:
                print('Final output:')
                print(', '.join(self.output))
            sys.exit(0)

    def _export_sdl(self, print_res=False):
        se, ne, nw, sw = compute_box(self.center, self.corner, self.box, MARGIN, self.inflation)
        r = sw - se
        u = ne - se
        lmin = 0.5 * (1 - 1 / self.inflation)
        lmax = 0.5 * (1 + 1 / self.inflation)

        lr = np.linalg.norm(ref_to_real_a(self.box, sw, MARGIN) -
                            ref_to_real_a(self.box, se, MARGIN))
        nr = int(ceil(lr / RESOLUTION))
        lu = np.linalg.norm(ref_to_real_a(self.box, ne, MARGIN) -
                            ref_to_real_a(self.box, se, MARGIN))
        nu = int(ceil(lu / RESOLUTION))

        if print_res:
            print('Mesh size: {}×{} points, {:.2f}×{:.2f} km²'.format(nr, nu, lr/1000, lu/1000))
            return

        ip = np.linspace(0, nr, nr+1) / nr
        jp = np.linspace(0, nu, nu+1) / nu
        xpts = se[0] + np.reshape(jp, (nr+1,1)) * r[0] + np.reshape(ip, (1,nu+1)) * u[0]
        ypts = se[1] + np.reshape(jp, (nr+1,1)) * r[1] + np.reshape(ip, (1,nu+1)) * u[1]
        xpts, ypts = ref_to_real_db(self.box, xpts, ypts, MARGIN)
        elevation = self.files.elevation(xpts, ypts)

        def bump(x):
            if lmin <= x <= lmax:
                return 1.0
            elif 0 < x < lmin:
                return exp(1-1.0/(1 - ((x-lmin)/lmin)**2))
            elif lmax < x < 1:
                return exp(1-1.0/(1 - ((x-lmax)/(1-lmax))**2))
            return 0.0

        rbump = np.array([bump(float(j)/nr) for j in range(nr+1)])
        ubump = np.array([bump(float(j)/nu) for j in range(nu+1)])
        elevation *= np.reshape(rbump, (nr+1,1))
        elevation *= np.reshape(ubump, (1,nu+1))

        data = np.zeros((nr, nu, 2), dtype=mesh.Mesh.dtype)
        p = Progress()
        for i in range(nr):
            for j in range(nu):
                data['vectors'][i,j,0] = np.array(
                    [[xpts[i,j], ypts[i,j], elevation[i,j]],
                     [xpts[i+1,j], ypts[i+1,j], elevation[i+1,j]],
                     [xpts[i,j+1], ypts[i,j+1], elevation[i,j+1]]])

                data['vectors'][i,j,1] = np.array(
                    [[xpts[i+1,j], ypts[i+1,j], elevation[i+1,j]],
                     [xpts[i+1,j+1], ypts[i+1,j+1], elevation[i+1,j+1]],
                     [xpts[i,j+1], ypts[i,j+1], elevation[i,j+1]]])

            p('Writing mesh: %i/%i' % (i+1, nr))
        p('Finished', end=True)

        data = np.reshape(data, (nr*nu*2,))
        m = mesh.Mesh(data, remove_empty_areas=False)

        fn = self._get_next_filename()
        m.save(fn)
        self.output.append(fn)
        print('Mesh size: {}×{} points, {:.2f}×{:.2f} km² => {}'.
              format(nr, nu, lr/1000, lu/1000, fn))

    def __call__(self, event):
        if isinstance(event, MouseEvent):
            self._handle_mouse(event)
        elif isinstance(event, KeyEvent):
            self._handle_key(event)

        self.ax.lines = []
        if self.center is not None and self.corner is not None:
            se, ne, nw, sw = compute_box(self.center, self.corner, self.box, MARGIN)
            self.ax.plot([se[0], ne[0], nw[0], sw[0], se[0]],
                         [se[1], ne[1], nw[1], sw[1], se[1]],
                         color='black', linestyle='--', linewidth=2.0)
            se, ne, nw, sw = compute_box(self.center, self.corner, self.box, MARGIN, self.inflation)
            self.ax.plot([se[0], ne[0], nw[0], sw[0], se[0]],
                         [se[1], ne[1], nw[1], sw[1], se[1]],
                         color='black', linestyle='--', linewidth=2.0)

        if self.center is not None:
            self.ax.plot([self.center[0]], [self.center[1]],
                         marker='o', markersize=8.0, color='black')
        if self.corner is not None:
            color = 'white' if self.inflation > 1.0 else 'red'
            self.ax.plot([self.corner[0]], [self.corner[1]],
                         marker='o', markersize=8.0, color=color)

        self.ax.figure.canvas.draw()


class DEMFiles:

    def __init__(self, *files):
        self.files = files

    def _compute_bounds(self):
        self.box = (min(f.east for f in self.files),
                    max(f.west for f in self.files),
                    min(f.south for f in self.files),
                    max(f.north for f in self.files))

        self.vmin = min(np.amin(f.data) for f in self.files)
        self.vmax = max(np.amax(f.data) for f in self.files)

    def elevation(self, xpts, ypts):
        elevation = np.zeros(xpts.shape, dtype=np.float32)
        found = np.zeros(xpts.shape, dtype=bool)

        for f in self.files:
            flags = np.logical_and(
                np.logical_and(f.east <= xpts, xpts <= f.west),
                np.logical_and(f.south <= ypts, ypts <= f.north),
            )
            found = np.logical_or(found, flags)

            i, j = np.nonzero(flags)
            elevation[i,j] = f.elevation(xpts[i,j], ypts[i,j])

        return elevation

    def show(self):
        self._compute_bounds()
        mpl.use('Qt5Agg')

        fig = plt.figure(figsize=(10,10))

        for f in self.files:
            f.plot(fig, self.box, self.vmin, self.vmax)

        overlay = fig.add_axes((0,0,1,1))
        overlay.axis('off')
        overlay.set_xlim(0, 1)
        overlay.set_ylim(0, 1)

        handler = ClickHandler(self, overlay, self.box)

        for event in ['button_press_event',
                      'scroll_event',
                      'key_press_event']:
            cid = fig.canvas.mpl_connect(event, handler)

        fig.show()
        # fig.savefig('out.png', dpi=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('dem.py')

    parser.add_argument('--resolution', '-r', required=False, default=10.0, type=float)
    parser.add_argument('--out', '-o', required=False, default='out.stl')
    parser.add_argument('--verbose', '-v', required=False, default=False,
                        action='store_true', help='Verbose')
    parser.add_argument('--store', metavar='fn', required=False, help='Store to HDF5 file')
    parser.add_argument('--hdf5', '-5', metavar='file', required=False, help='Get data from HDF5')
    parser.add_argument('files', metavar='file', nargs='+', help='DEM files or map IDs')

    args = parser.parse_args(sys.argv[1:])

    RESOLUTION = args.resolution
    OUTPUT = args.out

    if args.hdf5:
        with h5py.File(args.hdf5, 'r') as f:
            dems = [DEMFile(f, submap=fn, out=args.verbose) for fn in args.files]
    else:
        dems = [DEMFile(fn, out=args.verbose) for fn in args.files]

    if args.store:
        with h5py.File(args.store) as f:
            for d in dems:
                d.push_to_hdf5(f)
    else:
        files = DEMFiles(*dems)
        files.show()
        input()
