from __future__ import annotations

"""Линейная визуализация актуального local_moran_radius.geojson.
Полная замена управляющего скрипта линейного экспорта, без изменений src/.

НЕ пересчитывает Local Moran. Берег используется ТОЛЬКО для картографии.
Вход не обязан содержать segment_id, chainage_m, snap_ok и т.п.
Все исходные properties точек переносятся без изменений. Новые поля имеют
префикс render_. Сохраняются metadata входного экспорта и SHA-256 входов.

Каждая точка численно привязывается к ближайшей геодезической дуге WGS84.
На той же исходной части линии строится ячейка между серединами chainage
соседних привязанных точек. Для замкнутых частей используется кольцо.
Части без точек не окрашиваются. Несвязанные части не склеиваются.
MAX_HALF_SEGMENT_M ограничивает распространение цвета вдоль линии
(по умолчанию 5 км в каждую сторону). Это НЕ радиус статистического теста.
None: покрытие всей занятой части, как в старом экспортере.

GeoJSON всегда WGS84 lon/lat. Единой UTM нет. Пересечения антимеридиана
разрезаются. Длинные дуги дискретизируются OUTPUT_MAX_EDGE_M для GeoJSON.
Сегментация — условное отображение точечной статистики, не новый расчёт
статистики береговых сегментов и не пространственная интерполяция поля.

Зависимости: numpy, scipy, pyproj, geopandas, loguru; macOS/Linux: fcntl.
Временный GeoJSON проверяется и атомарно заменяет выход после проверки
неизменности входов. Входные файлы не изменяются.
"""

import fcntl
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import numpy as np
from loguru import logger
from pyproj import Geod, Transformer
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


def find_project_root():
    directory=Path(__file__).resolve().parent
    for candidate in (directory,*directory.parents):
        if (candidate/'src/waves/local_moran').is_dir():
            return candidate
    raise RuntimeError('Не найден корень BreachTheBeach: src/waves/local_moran.')


PROJECT_ROOT=find_project_root()
DATA_DIR=PROJECT_ROOT/'big_scale/data'
POINT_EXPORT_GEOJSON=DATA_DIR/'wave/local_moran_radius.geojson'
COASTLINE_GEOJSON=DATA_DIR/'S2CoastRussiaCropped.geojson'
OUTPUT_GEOJSON=DATA_DIR/'wave/local_moran_line_segments.geojson'

OVERWRITE_OUTPUT=False
MAX_SNAP_DISTANCE_M=100.0
STRICT_SNAP=False       # True: любая далёкая точка отменяет экспорт
MAX_HALF_SEGMENT_M=5000.0  # None: не ограничивать вдоль берега, до границ ячейки
DUPLICATE_CHAINAGE_TOLERANCE_M=0.01
FAIL_ON_DUPLICATE_CHAINAGE=False  # иначе оставить меньший point_id, записать список исключённых
MAX_INDEX_EDGE_M=10_000.0
MAX_INDEX_PIECES=5_000_000
SNAP_OPTIMIZATION_TOLERANCE_M=0.01
OUTPUT_MAX_EDGE_M=1000.0
LOG_EVERY=250
GEOJSON_INDENT=None
LOG_LEVEL='INFO'

GEOD=Geod(ellps='WGS84')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def resolve_path(value):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT/path).resolve()


def validate_settings():
    for name,value in [('MAX_SNAP_DISTANCE_M',MAX_SNAP_DISTANCE_M),
                       ('MAX_INDEX_EDGE_M',MAX_INDEX_EDGE_M),('OUTPUT_MAX_EDGE_M',OUTPUT_MAX_EDGE_M),
                       ('SNAP_OPTIMIZATION_TOLERANCE_M',SNAP_OPTIMIZATION_TOLERANCE_M),
                       ('DUPLICATE_CHAINAGE_TOLERANCE_M',DUPLICATE_CHAINAGE_TOLERANCE_M)]:
        require(math.isfinite(value) and value>0,f'{name} должен быть > 0.')
    require(MAX_INDEX_EDGE_M<=100_000,'MAX_INDEX_EDGE_M должен быть <= 100000 м.')
    require(OUTPUT_MAX_EDGE_M<=10_000,'OUTPUT_MAX_EDGE_M должен быть <= 10000 м.')
    if MAX_HALF_SEGMENT_M is not None:
        require(math.isfinite(MAX_HALF_SEGMENT_M) and MAX_HALF_SEGMENT_M>0,'Некорректный MAX_HALF_SEGMENT_M.')
    require(type(MAX_INDEX_PIECES) is int and MAX_INDEX_PIECES>0,'Некорректный MAX_INDEX_PIECES.')
    require(type(LOG_EVERY) is int and LOG_EVERY>0,'Некорректный LOG_EVERY.')
    for value in (OVERWRITE_OUTPUT,STRICT_SNAP,FAIL_ON_DUPLICATE_CHAINAGE):
        require(type(value) is bool,'Булевы параметры должны быть bool.')
    require(GEOJSON_INDENT is None or (type(GEOJSON_INDENT) is int and 0<=GEOJSON_INDENT<=8),'Некорректный GEOJSON_INDENT.')


@contextmanager
def file_lock(path):
    with path.open('a+',encoding='utf-8') as handle:
        try:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Файл занят: {path}') from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def state(path):
    s=path.stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns


def fingerprint(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_paths(inputs,output):
    protected=[*inputs,*[Path(f'{p}.lock') for p in inputs]]
    for a in protected:
        for b in (output,Path(f'{output}.lock')):
            same=a==b
            if a.exists() and b.exists(): same=same or a.samefile(b)
            require(not same,'Выход или его lock совпадает с входом.')


def reject_constant(value):
    raise ValueError(f'Недопустимое JSON-число {value}.')


def validate_numbers(value):
    if isinstance(value,float):
        require(math.isfinite(value),'Неконечное число в JSON.')
    elif isinstance(value,dict):
        for item in value.values(): validate_numbers(item)
    elif isinstance(value,list):
        for item in value: validate_numbers(item)


def load_points(path):
    with path.open(encoding='utf-8') as handle:
        data=json.load(handle,parse_constant=reject_constant)
    require(data.get('type')=='FeatureCollection','Ожидался точечный FeatureCollection.')
    crs=data.get('crs')
    if crs is not None:
        text=json.dumps(crs).lower()
        require('4326' in text or 'crs84' in text,'Входные точки должны быть WGS84 lon/lat.')
    points=[]; seen=set()
    for feature in data.get('features',[]):
        geom=feature.get('geometry') or {}; props=feature.get('properties') or {}
        require(geom.get('type')=='Point','Точечный экспорт содержит не-Point геометрию.')
        xy=geom.get('coordinates',[])
        require(len(xy)>=2 and all(type(v) in (int,float) and math.isfinite(v) for v in xy[:2]),'Неверные координаты.')
        lon,lat=float(xy[0]),float(xy[1])
        require(-180<=lon<=180 and -90<=lat<=90,'Координаты вне WGS84.')
        pid=props.get('point_id')
        require(type(pid) is int and pid not in seen,'point_id должен быть уникальным JSON-целым.')
        require(not any(str(k).startswith('render_') for k in props),'Префикс render_ зарезервирован этим экспортером.')
        require('normal_status' in props and 'alongshore_status' in props,'Нет полей актуального радиального экспорта.')
        validate_numbers(props); seen.add(pid)
        points.append(dict(pid=pid,lon=lon,lat=lat,props=props))
    require(bool(points),'Точечный экспорт пуст.')
    points.sort(key=lambda p:p['pid'])
    return points,data.get('metadata',{})


def iter_lines(geometry):
    if geometry is None or geometry.is_empty: return
    if geometry.geom_type=='LineString': yield geometry
    elif geometry.geom_type in {'MultiLineString','GeometryCollection'}:
        for part in geometry.geoms: yield from iter_lines(part)
    else:
        raise ValueError(f'Берег должен содержать линии, получено {geometry.geom_type}.')


class CoastPart:
    def __init__(self,part_id,coordinates):
        self.id=part_id
        xy=np.asarray(coordinates,dtype=float)[:,:2]
        require(len(xy)>=2 and np.isfinite(xy).all(),'Некорректные вершины берега.')
        require((np.abs(xy[:,0])<=180).all() and (np.abs(xy[:,1])<=90).all(),'Вершины берега вне WGS84.')
        # Последовательные физически совпадающие вершины не образуют рёбер.
        _,_,dist=GEOD.inv(xy[:-1,0],xy[:-1,1],xy[1:,0],xy[1:,1])
        xy=xy[np.r_[True,np.asarray(dist)>1e-7]]
        self.xy=xy
        if len(xy)<2:
            self.length=0.0
            return
        self.az,_,lengths=GEOD.inv(xy[:-1,0],xy[:-1,1],xy[1:,0],xy[1:,1])
        self.lengths=np.asarray(lengths,dtype=float)
        self.az=np.asarray(self.az)
        require(np.isfinite(self.lengths).all() and (self.lengths>0).all(),'Некорректные длины рёбер.')
        self.chain=np.r_[0.,np.cumsum(self.lengths)]
        self.length=float(self.chain[-1])
        self.closed=GEOD.inv(*xy[0],*xy[-1])[2]<1e-7

    def cut(self,start,end):
        require(0<=start<end<=self.length+1e-6,'Некорректные границы участка.')
        knots=np.r_[start,self.chain[(self.chain>start)&(self.chain<end)],end]
        samples=[]
        for left,right in zip(knots[:-1],knots[1:]):
            count=max(1,int(math.ceil((right-left)/OUTPUT_MAX_EDGE_M)))
            samples.extend(np.linspace(left,right,count+1)[:-1])
        samples=np.r_[samples,end]
        edge=np.clip(np.searchsorted(self.chain,samples,side='right')-1,0,len(self.lengths)-1)
        lon,lat,_=GEOD.fwd(self.xy[edge,0],self.xy[edge,1],self.az[edge],samples-self.chain[edge])
        return np.column_stack((lon,lat)).tolist()


class CoastIndex:
    def __init__(self,parts):
        self.parts=[]; arrays=[]; total=0
        self.ecef=Transformer.from_crs(4979,4978,always_xy=True)
        for part in parts:
            if part.length<=0: continue
            counts=np.ceil(part.lengths/MAX_INDEX_EDGE_M).astype(np.int64)
            count=int(counts.sum()); total+=count
            require(total<=MAX_INDEX_PIECES,'Превышен MAX_INDEX_PIECES; индекс слишком большой.')
            edge=np.repeat(np.arange(len(counts)),counts)
            offset=np.arange(count)-np.repeat(np.cumsum(counts)-counts,counts)
            lengths=part.lengths[edge]/counts[edge]
            distance=offset*lengths
            lon,lat,_=GEOD.fwd(part.xy[edge,0],part.xy[edge,1],part.az[edge],distance)
            ex,ey,_=GEOD.fwd(part.xy[edge,0],part.xy[edge,1],part.az[edge],distance+lengths)
            az,_,_=GEOD.inv(lon,lat,ex,ey)
            mx,my,_=GEOD.fwd(lon,lat,az,lengths/2)
            rank=len(self.parts); self.parts.append(part)
            arrays.append(np.column_stack((lon,lat,az,lengths,part.chain[edge]+distance,np.full(count,rank),mx,my)))
        require(bool(arrays),'Нет ненулевых береговых линий.')
        self.edges=np.concatenate(arrays)
        self.midpoints=np.column_stack(self.ecef.transform(self.edges[:,6],self.edges[:,7],np.zeros(total)))
        self.tree=cKDTree(self.midpoints)
        self.max_half=float(self.edges[:,3].max()/2)
        logger.info('Индекс берега: частей={}, рёбер={}',len(self.parts),len(self.edges))

    def edge_distance(self,index,lon,lat):
        row=self.edges[index]; length=float(row[3])
        def objective(s):
            x,y,_=GEOD.fwd(float(row[0]),float(row[1]),float(row[2]),float(s))
            return float(GEOD.inv(lon,lat,x,y)[2])
        opt=minimize_scalar(objective,bounds=(0.,length),method='bounded',
                            options={'xatol':SNAP_OPTIMIZATION_TOLERANCE_M})
        require(opt.success,'Не сошлась численная привязка к берегу.')
        return min((objective(0.),0.),(objective(length),length),(float(opt.fun),float(opt.x)))

    def nearest(self,lon,lat):
        xyz=np.asarray(self.ecef.transform(lon,lat,0.))
        seed=int(self.tree.query(xyz)[1]); best,along=self.edge_distance(seed,lon,lat); chosen=seed
        ids=np.asarray(self.tree.query_ball_point(xyz,best+self.max_half+0.1),dtype=np.int64)
        lower=np.linalg.norm(self.midpoints[ids]-xyz,axis=1)-self.edges[ids,3]/2
        for order in np.argsort(lower):
            if lower[order]>best+0.1: break
            index=int(ids[order])
            if index==seed: continue
            distance,s=self.edge_distance(index,lon,lat)
            if distance<best-1e-6 or (abs(distance-best)<=1e-6 and index<chosen):
                best,along,chosen=distance,s,index
        row=self.edges[chosen]; rank=int(row[5]); part=self.parts[rank]
        chain=min(part.length,max(0.,float(row[4])+along))
        if part.closed and part.length-chain<DUPLICATE_CHAINAGE_TOLERANCE_M: chain=0.
        return rank,chain,best


def cell_intervals(chainages,length,closed,cap):
    n=len(chainages); result=[]
    for i,current in enumerate(chainages):
        if closed:
            if n==1: left,right=current-length/2,current+length/2
            else:
                previous=chainages[i-1] if i else chainages[-1]-length
                following=chainages[i+1] if i<n-1 else chainages[0]+length
                left,right=(previous+current)/2,(current+following)/2
        else:
            left=0. if i==0 else (chainages[i-1]+current)/2
            right=length if i==n-1 else (current+chainages[i+1])/2
        if cap is not None:
            left=max(left,current-cap); right=min(right,current+cap)
        span=right-left
        if span<=1e-7: result.append([]); continue
        if not closed:
            result.append([(max(0.,left),min(length,right))]); continue
        if span>=length-1e-7:
            result.append([(0.,length)]); continue
        left=left%length; end=left+span
        result.append([(left,end)] if end<=length else [(left,length),(0.,end-length)])
    return result


def split_antimeridian(coords):
    parts=[]; current=[coords[0]]
    for a,b in zip(coords[:-1],coords[1:]):
        delta=b[0]-a[0]
        if abs(delta)<=180:
            current.append(b); continue
        direction=1 if delta<0 else -1
        boundary=180. if direction>0 else -180.
        az,_,distance=GEOD.inv(*a,*b)
        lo,hi=0.,distance
        for _ in range(50):
            mid=(lo+hi)/2; x,y,_=GEOD.fwd(*a,az,mid)
            unwrapped=a[0]+((x-a[0]+180)%360-180)
            if (unwrapped<boundary if direction>0 else unwrapped>boundary): lo=mid
            else: hi=mid
        _,y,_=GEOD.fwd(*a,az,(lo+hi)/2)
        crossing=[boundary,float(y)]
        if current[-1]!=crossing: current.append(crossing)
        if len(current)>=2 and any(p!=current[0] for p in current[1:]): parts.append(current)
        current=[[-boundary,float(y)]]
        if current[-1]!=b: current.append(b)
    if len(current)>=2 and any(p!=current[0] for p in current[1:]): parts.append(current)
    return parts


def validate_geometry(geometry):
    arrays=[geometry['coordinates']] if geometry['type']=='LineString' else geometry['coordinates']
    require(bool(arrays),'Пустая линейная геометрия.')
    for coordinates in arrays:
        require(len(coordinates)>=2,'В линии меньше двух вершин.')
        for lon,lat in coordinates:
            require(math.isfinite(lon) and math.isfinite(lat) and -180<=lon<=180 and -90<=lat<=90,'Некорректные выходные координаты.')
        require(all(abs(a[0]-b[0])<=180+1e-8 for a,b in zip(coordinates[:-1],coordinates[1:])),
                'Неразрезанное пересечение антимеридиана.')


def build_features(points,index):
    groups=defaultdict(list); unmatched=[]
    for number,point in enumerate(points,1):
        rank,chain,distance=index.nearest(point['lon'],point['lat'])
        if distance>MAX_SNAP_DISTANCE_M:
            unmatched.append(dict(point_id=point['pid'],snap_distance_m=distance))
        else:
            groups[rank].append(dict(point=point,chain=chain,distance=distance))
        if number%LOG_EVERY==0 or number==len(points):
            logger.info('Картографическая привязка: {}/{}',number,len(points))
    require(not unmatched or not STRICT_SNAP,f'За пределами MAX_SNAP_DISTANCE_M: {unmatched[:10]}')
    if unmatched: logger.warning('Не отображены далёкие от берега точки: {}',len(unmatched))
    duplicates=[]; records=[]
    for rank,rows in sorted(groups.items()):
        part=index.parts[rank]
        rows.sort(key=lambda r:(r['chain'],r['point']['pid']))
        clusters=[]
        for row in rows:
            if not clusters or row['chain']-clusters[-1][0]['chain']>DUPLICATE_CHAINAGE_TOLERANCE_M:
                clusters.append([row])
            else: clusters[-1].append(row)
        unique=[]
        for cluster in clusters:
            chosen=min(cluster,key=lambda r:r['point']['pid'])
            unique.append(chosen)
            duplicates.extend(r['point']['pid'] for r in cluster if r is not chosen)
        require(not duplicates or not FAIL_ON_DUPLICATE_CHAINAGE,
                f'Неоднозначная привязка совпадающих точек: {duplicates[:20]}')
        unique.sort(key=lambda r:r['chain'])
        bounds=cell_intervals([r['chain'] for r in unique],part.length,part.closed,MAX_HALF_SEGMENT_M)
        for row,intervals in zip(unique,bounds):
            pieces=[]; length=0.
            for start,end in intervals:
                if end-start<=1e-7: continue
                pieces.extend(split_antimeridian(part.cut(start,end))); length+=end-start
            if not pieces: continue
            geometry=dict(type='LineString',coordinates=pieces[0]) if len(pieces)==1 else dict(type='MultiLineString',coordinates=pieces)
            validate_geometry(geometry)
            point=row['point']
            props={**point['props'],
                   'render_segment_id':f"{part.id}_point_{point['pid']}",
                   'render_source_part':part.id,'render_chainage_m':row['chain'],
                   'render_snap_distance_m':row['distance'],'render_length_m':length,
                   'render_part_length_m':part.length,'render_part_closed':part.closed,
                   'render_intervals_m':[[float(a),float(b)] for a,b in intervals]}
            records.append(dict(type='Feature',id=point['pid'],geometry=geometry,properties=props))
    require(bool(records),'Не удалось построить ни одного линейного участка.')
    if duplicates: logger.warning('Совпадающие привязки: {} точек не показаны; приоритет меньшего point_id.',len(duplicates))
    stats=dict(input_points=len(points),output_features=len(records),matched_parts=len(groups),
               unmatched_points=unmatched,duplicate_point_ids=sorted(duplicates))
    return records,stats


def main():
    logger.remove(); logger.add(sys.stderr,level=LOG_LEVEL,diagnose=False,backtrace=False)
    validate_settings()
    point_file,coast_file,output=map(resolve_path,(POINT_EXPORT_GEOJSON,COASTLINE_GEOJSON,OUTPUT_GEOJSON))
    for path in (point_file,coast_file):
        require(path.is_file(),f'Не найден файл: {path}')
    output.parent.mkdir(parents=True,exist_ok=True)
    check_paths([point_file,coast_file],output)
    started=time.monotonic()
    with ExitStack() as stack:
        for path in sorted({point_file,coast_file,output},key=str):
            stack.enter_context(file_lock(Path(f'{path}.lock')))
        check_paths([point_file,coast_file],output)
        require(not output.exists() or OVERWRITE_OUTPUT,f'Выход существует: {output}. Для замены OVERWRITE_OUTPUT=True.')
        states={path:state(path) for path in (point_file,coast_file)}
        hashes={str(path):fingerprint(path) for path in (point_file,coast_file)}
        points,source_metadata=load_points(point_file)
        import geopandas as gpd
        frame=gpd.read_file(coast_file)
        require(not frame.empty and frame.crs is not None,'Берег пуст или не имеет CRS.')
        coastline_crs=frame.crs.to_string(); frame=frame.to_crs(4326)
        def parts():
            for feature_id,geometry in enumerate(frame.geometry):
                for part_id,line in enumerate(iter_lines(geometry)):
                    yield CoastPart(f'feature_{feature_id:06d}_part_{part_id:04d}',line.coords)
        index=CoastIndex(parts())
        features,stats=build_features(points,index)
        metadata=dict(created_utc=datetime.now(UTC).isoformat(),
                      purpose='cartographic transfer only; radial Local Moran unchanged',
                      source_point_metadata=source_metadata,input_sha256=hashes,coastline_crs=coastline_crs,
                      distance_method='WGS84 geodesic; no common projected CRS',
                      assignment='nearest original coastline part, midpoint chainage cells',
                      max_snap_distance_m=MAX_SNAP_DISTANCE_M,max_half_segment_m=MAX_HALF_SEGMENT_M,
                      snap_optimization_tolerance_m=SNAP_OPTIMIZATION_TOLERANCE_M,
                      duplicate_chainage_tolerance_m=DUPLICATE_CHAINAGE_TOLERANCE_M,
                      duplicate_policy='fail' if FAIL_ON_DUPLICATE_CHAINAGE else 'lowest_point_id',
                      output_max_edge_m=OUTPUT_MAX_EDGE_M,stats=stats)
        validate_numbers(metadata)
        with TemporaryDirectory(prefix=f'.{output.stem}.',dir=output.parent) as directory:
            temporary=Path(directory)/'lines.geojson'
            with temporary.open('w',encoding='utf-8') as handle:
                json.dump(dict(type='FeatureCollection',metadata=metadata,features=features),handle,
                          ensure_ascii=False,allow_nan=False,indent=GEOJSON_INDENT)
                handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
            for path,expected in states.items():
                require(state(path)==expected and fingerprint(path)==hashes[str(path)],
                        f'Вход изменился во время работы, выход не опубликован: {path}')
            os.replace(temporary,output)
    logger.success('Готово: линий={}, частей с точками={}, {:.1f} с, {}',
                   len(features),stats['matched_parts'],time.monotonic()-started,output)


if __name__=='__main__':
    main()
