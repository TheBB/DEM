import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

MARGIN = 0.02
INFLATION = 1.5

def real_to_ref(min, max, x, margin):
    return margin + (1.0 - 2*margin) * (x - min) / (max - min)

def real_to_ref_a(box, x, margin):
    return np.array((real_to_ref(box[0], box[1], x[0], margin),
                     real_to_ref(box[2], box[3], x[1], margin)))

def ref_to_real(min, max, x, margin):
    result = (x - margin) / (1.0 - 2*margin) * (max - min) + min
    return result

def ref_to_real_a(box, x, margin):
    return np.array((ref_to_real(box[0], box[1], x[0], margin),
                     ref_to_real(box[2], box[3], x[1], margin)))

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

class DEMFile:
    def __init__(self, fn, out=False):
        with open(fn, 'rb') as f:
            s = f.read()
        self._load_record_a(s[:1024], out)
        self._check()
        self.data = []

        init = 1024
        for _ in range(0, self.size):
            init = self._load_data(s, init)

        self.data = np.rot90(np.array(self.data))

    def plot(self, fig, box, vmin, vmax):
        east = real_to_ref(box[0], box[1], self.east, MARGIN)
        west = real_to_ref(box[0], box[1], self.west, MARGIN)
        south = real_to_ref(box[2], box[3], self.south, MARGIN)
        north = real_to_ref(box[2], box[3], self.north, MARGIN)

        ax = fig.add_axes((east, south, west-east, north-south))
        ax.imshow(self.data, cmap='bwr', aspect='auto',
                  extent=(self.east, self.west,
                          self.south, self.north),
                  vmin=vmin, vmax=vmax)
        ax.axis('off')

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

        data = [int(v)*self.resolution[2] + params['local']
                for v in s[init+144:init+1024].decode().split()]
        while len(data) < size:
            init += 1024
            data.extend(int(v)*self.resolution[2] + params['local']
                        for v in s[init:init+1024].decode().split())

        assert(len(data) == size)

        # print('Found profile at {} with {} asserted and {} real data points ({:.2f} to {:.2f})'.format(
        #     pos, size, len(data), min(data), max(data)))
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
        assert self.elevations[0] < self.elevations[1]
        assert self.reference_angle == 0.0
        assert len(self.size) == 2
        assert self.size[0] == 1

        self.size = self.size[1]
        self.east = self.corners[0]
        self.west = self.corners[4]
        self.south = self.corners[1]
        self.north = self.corners[3]


class ClickHandler:

    def __init__(self, ax, box):
        self.center = None
        self.corner = None
        self.ax = ax
        self.box = box

    def __call__(self, event):
        if event.button == 1:
            self.center = np.array((event.xdata, event.ydata))
        elif event.button == 3:
            self.corner = np.array((event.xdata, event.ydata))

        self.ax.lines = []
        if self.center is not None:
            self.ax.plot([self.center[0]], [self.center[1]], marker='o', color='green')
        if self.corner is not None:
            self.ax.plot([self.corner[0]], [self.corner[1]], marker='o', color='yellow')
        if self.center is not None and self.corner is not None:
            se, ne, nw, sw = compute_box(self.center, self.corner, self.box, MARGIN)
            self.ax.plot([se[0], ne[0], nw[0], sw[0], se[0]],
                         [se[1], ne[1], nw[1], sw[1], se[1]],
                         color='yellow')

            se, ne, nw, sw = compute_box(self.center, self.corner, self.box, MARGIN, INFLATION)
            self.ax.plot([se[0], ne[0], nw[0], sw[0], se[0]],
                         [se[1], ne[1], nw[1], sw[1], se[1]],
                         color='orange')

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

    def show(self):
        self._compute_bounds()
        mpl.use('Qt5Agg')

        fig = plt.figure()

        for f in self.files:
            f.plot(fig, self.box, self.vmin, self.vmax)
        # ax = fig.add_axes((0.02,0.02,0.96,0.96))
        # self.files[0].plot(ax)

        overlay = fig.add_axes((0,0,1,1))
        overlay.axis('off')
        overlay.set_xlim(0, 1)
        overlay.set_ylim(0, 1)

        handler = ClickHandler(overlay, self.box)

        cid = fig.canvas.mpl_connect('button_press_event', handler)

        fig.show()


# dem = DEMFile('7002_2_10m_z33.dem', True)
# files = DEMFiles(dem)
# files.show()

dem1 = DEMFile('6904_1_10m_z32.dem')
print 'Loaded 1'
dem2 = DEMFile('6904_2_10m_z32.dem')
print 'Loaded 2'
dem3 = DEMFile('6904_3_10m_z32.dem')
print 'Loaded 3'
dem4 = DEMFile('6904_4_10m_z32.dem')
print 'Loaded 4'
files = DEMFiles(dem1, dem2, dem3, dem4)
files.show()

input()
