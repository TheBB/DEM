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
    def __init__(self, s, out=True):
        self._load_record_a(s[:1024], out)
        self._check()

        init = 1024
        for _ in range(0, self.size):
            init = self._load_data(s, init)

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

        print('Found profile at {} with {} asserted and {} real data points ({:.2f} to {:.2f})'.format(
            pos, size, len(data), min(data), max(data)))

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


with open('6904_2_10m_z32.dem', 'rb') as f:
    s = f.read()
recA = DEMFile(s)
