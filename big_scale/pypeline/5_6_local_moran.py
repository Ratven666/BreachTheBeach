from __future__ import annotations

"""Радиальный Local Moran по точкам WGS84, без береговой трассы.
Полная замена big_scale/pypeline/5_6_local_moran.py. Файлы src/ не меняются.

Соседи: все ДРУГИЕ точки с Geod(WGS84).inv distance <= RADIUS_M.
Острова, проливы, противоположные берега и суша не являются барьерами.
ECEF/cKDTree только отбирает кандидатов; радиус проверяется на эллипсоиде.

Это пространственная автокорреляция значений из point_trends, НЕ Pearson
между временными рядами. Источник: normal_mean_wm и alongshore_abs/net_mean_wm.
Стандартизация отдельно по каждой метрике по всем её конечным значениям:
 z=(x-mean)/sqrt(mean((x-mean)**2)); I_i=z_i*sum(w_ij*z_j), sum_j(w_ij)=1.

Условные перестановки: значение центральной точки фиксировано, k значений
без возвращения из ВСЕХ остальных пригодных точек (не только из радиуса).
Тест двусторонний относительно аналитического E[I_i]=-z_i**2/(n_valid-1):
 p=(1+count(abs(I_perm-E)>=abs(I_obs-E)))/(permutations+1).
Не обещается совпадение p-value с PySAL: определения хвостов могут отличаться.
BH-FDR отдельно по метрике среди невырожденных тестов status=OK.
При вырождении p/q=NULL; квадрант HH/LL/HL/LH остаётся описательным.

Новые независимые таблицы: radius_moran_runs / radius_moran_points /
 radius_moran_results. Старые local_moran_* НЕ меняются.
Каждый запуск сохраняет координаты, исходные значения и параметры.
Публикация SQLite — одна транзакция без полной копии БД.
GeoJSON публикуется отдельно атомарной заменой файла. Если экспорт не удался,
запуск остаётся в БД; для повторного экспорта MODE='export'.

Зависимости проекта: numpy, scipy, pyproj, loguru; macOS/Linux: fcntl.
Береговой GeoJSON, GeoPandas, UTM, WINDOW_WIDTH_M и snap не нужны.
"""

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
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
from scipy.spatial import cKDTree


def find_project_root():
    directory=Path(__file__).resolve().parent
    for candidate in (directory,*directory.parents):
        if (candidate/'src/waves/activity_indices/service.py').is_file():
            return candidate
    raise RuntimeError('Не найден корень BreachTheBeach.')


PROJECT_ROOT=find_project_root()
DATA_DIR=PROJECT_ROOT/'big_scale/data'
MODE='calculate_export'  # calculate | export | calculate_export
INDEX_DB=DATA_DIR/'db/wave_activity_indices.db'
OUTPUT_GEOJSON=DATA_DIR/'wave/local_moran_radius.geojson'

RADIUS_M=1_600.0
ALONGSHORE_MODE='abs'  # abs | net
WEIGHT_MODE='binary'  # binary | inverse_distance; оба нормируются по строке
MIN_NEIGHBORS=1       # число ДРУГИХ точек с пригодной метрикой
MIN_DISTANCE_M=1.0    # нижняя граница для 1/d, в том числе совпадающих координат
PERMUTATIONS=999
RANDOM_SEED=42
ALPHA_RAW=0.999
ALPHA_FDR=0.999
EXPORT_RUN_ID=None    # None: последний радиальный запуск; calculate_export: новый
OVERWRITE_GEOJSON=False
GEOJSON_INDENT=None
FULL_INTEGRITY_CHECK=False
MAX_DIRECTED_EDGES=10_000_000  # защита RAM от почти полного графа
LOG_EVERY=250
LOG_LEVEL='INFO'

METHOD='radius_geodesic_global_z_conditional_two_sided_v1'
RESULT_FIELDS=('point_id','metric','value','n_valid','n_neighbors_total','n_neighbors',
               'reference_mean','reference_sd','neighbor_mean','mean_neighbor_distance_m',
               'max_neighbor_distance_m','z_value','spatial_lag','local_i',
               'permutation_expected_i','permutation_sd','p_sim','q_fdr_bh',
               'quadrant','cluster','cluster_fdr','status')
DDL=(
'''CREATE TABLE IF NOT EXISTS radius_moran_runs (
 run_id INTEGER PRIMARY KEY AUTOINCREMENT,created_utc TEXT NOT NULL,
 method TEXT NOT NULL,config_json TEXT NOT NULL,source_meta_json TEXT NOT NULL,
 source_data_sha256 TEXT NOT NULL,n_points INTEGER NOT NULL,n_results INTEGER NOT NULL)''',
'''CREATE TABLE IF NOT EXISTS radius_moran_points (
 run_id INTEGER NOT NULL,point_id INTEGER NOT NULL,lon REAL NOT NULL,lat REAL NOT NULL,
 PRIMARY KEY(run_id,point_id),FOREIGN KEY(run_id) REFERENCES radius_moran_runs(run_id) ON DELETE CASCADE)''',
'''CREATE TABLE IF NOT EXISTS radius_moran_results (
 run_id INTEGER NOT NULL,point_id INTEGER NOT NULL,metric TEXT NOT NULL,value REAL,
 n_valid INTEGER NOT NULL,n_neighbors_total INTEGER NOT NULL,n_neighbors INTEGER NOT NULL,
 reference_mean REAL,reference_sd REAL,neighbor_mean REAL,mean_neighbor_distance_m REAL,
 max_neighbor_distance_m REAL,z_value REAL,spatial_lag REAL,local_i REAL,
 permutation_expected_i REAL,permutation_sd REAL,p_sim REAL CHECK(p_sim BETWEEN 0 AND 1),
 q_fdr_bh REAL CHECK(q_fdr_bh BETWEEN 0 AND 1),quadrant TEXT,cluster TEXT,cluster_fdr TEXT,status TEXT NOT NULL,
 PRIMARY KEY(run_id,point_id,metric),
 FOREIGN KEY(run_id,point_id) REFERENCES radius_moran_points(run_id,point_id) ON DELETE CASCADE)''',
)


def require(condition,message):
    if not condition:
        raise ValueError(message)


def path_resolve(value):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT/path).resolve()


def validate_settings():
    require(MODE in {'calculate','export','calculate_export'},'Некорректный MODE.')
    require(type(OVERWRITE_GEOJSON) is bool and type(FULL_INTEGRITY_CHECK) is bool,'Некорректный bool.')
    require(EXPORT_RUN_ID is None or (type(EXPORT_RUN_ID) is int and EXPORT_RUN_ID>0),'Некорректный EXPORT_RUN_ID.')
    require(GEOJSON_INDENT is None or (type(GEOJSON_INDENT) is int and 0<=GEOJSON_INDENT<=8),'Некорректный GEOJSON_INDENT.')
    if MODE=='export':
        return
    require(math.isfinite(RADIUS_M) and RADIUS_M>0,'RADIUS_M должен быть > 0.')
    require(ALONGSHORE_MODE in {'abs','net'},'ALONGSHORE_MODE: abs | net.')
    require(WEIGHT_MODE in {'binary','inverse_distance'},'Некорректный WEIGHT_MODE.')
    require(math.isfinite(MIN_DISTANCE_M) and MIN_DISTANCE_M>0,'MIN_DISTANCE_M должен быть > 0.')
    for name,value,minimum in [('MIN_NEIGHBORS',MIN_NEIGHBORS,1),('PERMUTATIONS',PERMUTATIONS,99),
                               ('RANDOM_SEED',RANDOM_SEED,0),('MAX_DIRECTED_EDGES',MAX_DIRECTED_EDGES,1),
                               ('LOG_EVERY',LOG_EVERY,1)]:
        require(type(value) is int and value>=minimum,f'{name} должен быть целым >= {minimum}.')
    for name,value in [('ALPHA_RAW',ALPHA_RAW),('ALPHA_FDR',ALPHA_FDR)]:
        require(math.isfinite(value) and 0<value<1,f'{name} должен быть в (0,1).')
        if value>0.1:
            logger.warning('{}={}: очень мягкий порог; для всех квадрантов используй quadrant.',name,value)


@contextmanager
def file_lock(path):
    with path.open('a+',encoding='utf-8') as handle:
        try:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Файл занят другим этапом: {path}') from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def readonly(path):
    con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=60,isolation_level=None)
    con.execute('PRAGMA query_only=ON')
    return con


def identity(path):
    stat=path.stat()
    return stat.st_dev,stat.st_ino


def version(con):
    return int(con.execute('PRAGMA data_version').fetchone()[0])


def guard(watcher,expected,path,inode):
    require(identity(path)==inode and version(watcher)==expected,
            'Исходная БД изменилась во время расчёта. Результат не опубликован; повтори запуск без параллельной записи.')


def check_paths(database,output):
    require(output.suffix.lower() in {'.json','.geojson'},'Выход должен быть GeoJSON.')
    for protected in [database]+[Path(f'{database}{s}') for s in ('-wal','-shm','-journal','.lock')]:
        for candidate in (output,Path(f'{output}.lock')):
            same=protected==candidate
            if protected.exists() and candidate.exists():
                same=same or protected.samefile(candidate)
            require(not same,'Выход или его lock совпадает с файлом БД.')


def load_source(database):
    con=readonly(database)
    try:
        con.execute('BEGIN')
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require({'points','point_trends','meta'}<=tables,'Нужна wave_activity_indices.db из шага 5.4.')
        if FULL_INTEGRITY_CHECK:
            logger.info('Полная проверка исходной БД...')
            for row in con.execute('PRAGMA quick_check'):
                require(row[0]=='ok',f'SQLite: {row[0]}')
            require(con.execute('PRAGMA foreign_key_check').fetchone() is None,'Нарушены FK исходной БД.')
        metric='alongshore_abs_mean_wm' if ALONGSHORE_MODE=='abs' else 'alongshore_net_mean_wm'
        rows=con.execute(f'''SELECT p.point_id,p.lon,p.lat,t.normal_mean_wm,t.{metric},t.point_id
            FROM points p LEFT JOIN point_trends t ON t.point_id=p.point_id ORDER BY p.point_id''').fetchall()
        require(bool(rows),'Нет точек.')
        require(all(r[5] is not None for r in rows),'Не для всех точек есть point_trends.')
        points=[]
        for pid,lon,lat,normal,along,_ in rows:
            lon,lat=float(lon),float(lat)
            require(math.isfinite(lon) and math.isfinite(lat) and -180<=lon<=180 and -90<=lat<=90,
                    f'Некорректные координаты point_id={pid}.')
            vals=[]
            for value in (normal,along):
                vals.append(None if value is None or not math.isfinite(float(value)) else float(value))
            points.append((int(pid),lon,lat,*vals))
        require(len({p[0] for p in points})==len(points),'Повторные point_id.')
        meta=dict(con.execute('SELECT key,value FROM meta'))
        gaps=int(meta.get('source_missing_bathymetry_point_days','0'))
        if gaps:
            logger.warning('Источник содержит {} точко-дней с пропусками батиметрии. Они не исправляются.',gaps)
        digest=hashlib.sha256(json.dumps(points,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        return points,['normal_mean_wm',metric],meta,digest
    finally:
        con.close()


def radius_neighbors(points,radius,max_edges):
    xy=np.asarray([(p[1],p[2]) for p in points],dtype=float)
    transformer=Transformer.from_crs(4979,4978,always_xy=True)
    xyz=np.column_stack(transformer.transform(xy[:,0],xy[:,1],np.zeros(len(points))))
    tree=cKDTree(xyz)
    geod=Geod(ellps='WGS84')
    neighbors=[]
    edges=0
    for i,(lon,lat) in enumerate(xy):
        # Хорда <= геодезического расстояния, поэтому кандидаты не теряются.
        candidate=np.asarray(tree.query_ball_point(xyz[i],radius+1e-4),dtype=np.int64)
        candidate=candidate[candidate!=i]  # совпадающие координаты других ID допускаются
        candidate.sort()
        if len(candidate):
            distances=np.asarray(geod.inv(np.full(len(candidate),lon),np.full(len(candidate),lat),
                                          xy[candidate,0],xy[candidate,1])[2])
            mask=distances<=radius+1e-6
            candidate,distances=candidate[mask],distances[mask]
        else:
            distances=np.empty(0,dtype=float)
        edges+=len(candidate)
        require(edges<=max_edges,'Превышен MAX_DIRECTED_EDGES: граф слишком плотный. Уменьши радиус или осознанно увеличь лимит RAM.')
        neighbors.append((candidate,distances))
        if (i+1)%LOG_EVERY==0 or i+1==len(points):
            logger.info('Соседство: {}/{} точек; направленных связей={}',i+1,len(points),edges)
    counts=np.asarray([len(a) for a,_ in neighbors])
    logger.info('Радиус {} м: без соседей={}, min/median/max={}/{}/{}',radius,
                int(np.count_nonzero(counts==0)),int(counts.min()),float(np.median(counts)),int(counts.max()))
    return neighbors


def quadrant(z,lag):
    if abs(z)<=1e-14 or abs(lag)<=1e-14:
        return 'UNDEFINED'
    return ('H' if z>0 else 'L')+('H' if lag>0 else 'L')


def apply_fdr(rows,alpha):
    valid=[i for i,r in enumerate(rows) if r['status']=='OK' and r['p_sim'] is not None]
    if not valid:
        return
    order=np.argsort([rows[i]['p_sim'] for i in valid],kind='stable')
    p=np.asarray([rows[valid[i]]['p_sim'] for i in order])
    q=np.minimum.accumulate((p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    for position,value in zip(order,np.clip(q,0,1)):
        row=rows[valid[int(position)]]
        row['q_fdr_bh']=float(value)
        row['cluster_fdr']=row['quadrant'] if value<=alpha and row['quadrant']!='UNDEFINED' else 'NS'


def compute_metric(points,neighbors,metric,column,metric_code):
    values=np.asarray([np.nan if p[column] is None else p[column] for p in points],dtype=float)
    finite=np.isfinite(values)
    valid_idx=np.flatnonzero(finite)
    n=len(valid_idx)
    mean=float(values[finite].mean()) if n else None
    sd=float(values[finite].std(ddof=0)) if n else None
    usable=n>=3 and sd is not None and sd>0 and math.isfinite(sd)
    z=np.full(len(points),np.nan)
    if usable:
        z[finite]=(values[finite]-mean)/sd
        require(np.isfinite(z[finite]).all(),'Численная ошибка стандартизации метрики.')
    rows=[]
    for i,point in enumerate(points):
        idx,dist=neighbors[i]
        keep=finite[idx]
        idx,dist=idx[keep],dist[keep]
        row={key:None for key in RESULT_FIELDS}
        row.update(point_id=point[0],metric=metric,value=None if not finite[i] else float(values[i]),
                   n_valid=n,n_neighbors_total=len(neighbors[i][0]),n_neighbors=len(idx),
                   reference_mean=mean,reference_sd=sd)
        if len(idx):
            weights=np.ones(len(idx)) if WEIGHT_MODE=='binary' else 1.0/np.maximum(dist,MIN_DISTANCE_M)
            weights/=weights.sum()
            row.update(neighbor_mean=float(np.dot(weights,values[idx])),
                       mean_neighbor_distance_m=float(dist.mean()),max_neighbor_distance_m=float(dist.max()))
        if not finite[i]:
            status='MISSING_VALUE'
        elif n<3:
            status='INSUFFICIENT_DATA'
        elif not usable:
            status='NO_VARIATION'
        elif len(idx)<MIN_NEIGHBORS:
            status='NO_NEIGHBORS' if not len(idx) else 'INSUFFICIENT_NEIGHBORS'
        else:
            zi=float(z[i]); lag=float(np.dot(weights,z[idx])); observed=zi*lag
            expected=-zi*zi/(n-1)
            row.update(z_value=zi,spatial_lag=lag,local_i=observed,
                       permutation_expected_i=expected,quadrant=quadrant(zi,lag))
            pool=z[valid_idx[valid_idx!=i]]
            m=len(pool)
            variance_pool=float(np.var(pool))
            factor=(m*float(np.dot(weights,weights))-1)/(m-1)
            if zi==0 or variance_pool==0 or factor<=64*np.finfo(float).eps:
                status='DEGENERATE_PERMUTATION'
            else:
                uid=int(point[0]) & ((1<<64)-1)
                rng=np.random.default_rng(np.random.SeedSequence([RANDOM_SEED,uid&0xffffffff,uid>>32,metric_code]))
                simulations=np.empty(PERMUTATIONS)
                for b in range(PERMUTATIONS):
                    chosen=rng.choice(pool,size=len(idx),replace=False)
                    simulations[b]=zi*float(np.dot(weights,chosen))
                sim_sd=float(np.std(simulations,ddof=1))
                row['permutation_sd']=sim_sd
                if sim_sd==0:
                    status='DEGENERATE_SIMULATION'
                else:
                    tolerance=64*np.finfo(float).eps*max(1.0,abs(observed),abs(expected))
                    exceed=np.count_nonzero(np.abs(simulations-expected)>=abs(observed-expected)-tolerance)
                    row['p_sim']=float((1+exceed)/(PERMUTATIONS+1))
                    row['cluster']=row['quadrant'] if row['p_sim']<=ALPHA_RAW and row['quadrant']!='UNDEFINED' else 'NS'
                    row['cluster_fdr']='NS'
                    status='OK'
        row['status']=status
        if status!='OK':
            row['cluster']=row['cluster_fdr']=status
        rows.append(row)
        if (i+1)%LOG_EVERY==0 or i+1==len(points):
            logger.info('{}: {}/{} точек',metric,i+1,len(points))
    apply_fdr(rows,ALPHA_FDR)
    logger.info('{} status: {}',metric,dict(Counter(r['status'] for r in rows)))
    logger.info('{} FDR clusters: {}',metric,dict(Counter(r['cluster_fdr'] for r in rows)))
    return rows


def current_config(metrics):
    return dict(radius_m=RADIUS_M,distance='WGS84 ellipsoidal geodesic',neighbor_rule='all other point IDs within radius; no barriers',
                metrics=metrics,weight_mode=WEIGHT_MODE,row_standardized=True,min_neighbors=MIN_NEIGHBORS,
                inverse_distance_floor_m=MIN_DISTANCE_M,permutations=PERMUTATIONS,seed=RANDOM_SEED,
                alpha_raw=ALPHA_RAW,alpha_fdr=ALPHA_FDR,standardization='global per metric, population SD',
                local_i='z_i * row_standardized_spatial_lag',permutation_pool='all other finite points per metric',
                p_value='two-sided about analytic conditional expectation; inclusive ties; plus-one correction',
                fdr_family='separately per metric, status=OK',max_directed_edges=MAX_DIRECTED_EDGES)


def validate_results(points,metrics,results):
    expected={(p[0],m) for p in points for m in metrics}
    actual={(r['point_id'],r['metric']) for r in results}
    require(actual==expected and len(results)==len(expected),'Неполный набор результатов.')
    for row in results:
        require(all(not isinstance(v,float) or math.isfinite(v) for v in row.values()),'Неконечные числа в результате.')
        for key in ('p_sim','q_fdr_bh'):
            require(row[key] is None or 0<=row[key]<=1,f'Некорректный {key}.')
        require(row['status']!='OK' or (row['p_sim'] is not None and row['q_fdr_bh'] is not None),
                'Строка OK без p/q.')


def publish(database,points,metrics,results,meta,digest,check):
    validate_results(points,metrics,results)
    con=sqlite3.connect(database.as_uri()+'?mode=rw',uri=True,timeout=60,isolation_level=None)
    try:
        con.execute('PRAGMA foreign_keys=ON'); con.execute('PRAGMA synchronous=FULL')
        con.execute('BEGIN IMMEDIATE')
        try:
            check()
            for ddl in DDL:
                con.execute(ddl)
            cursor=con.execute('''INSERT INTO radius_moran_runs
                (created_utc,method,config_json,source_meta_json,source_data_sha256,n_points,n_results)
                VALUES (?,?,?,?,?,?,?)''',(datetime.now(UTC).isoformat(),METHOD,
                json.dumps(current_config(metrics),allow_nan=False,ensure_ascii=False),
                json.dumps(meta,allow_nan=False,ensure_ascii=False),digest,len(points),len(results)))
            run_id=int(cursor.lastrowid)
            con.executemany('INSERT INTO radius_moran_points VALUES (?,?,?,?)',
                            ((run_id,p[0],p[1],p[2]) for p in points))
            fields=('run_id',)+RESULT_FIELDS
            con.executemany(f"INSERT INTO radius_moran_results ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                            ((run_id,)+(tuple(r[k] for k in RESULT_FIELDS)) for r in results))
            con.execute('COMMIT')
        except BaseException:
            if con.in_transaction:
                con.execute('ROLLBACK')
            raise
        return run_id
    finally:
        con.close()


def load_run(database,run_id):
    con=readonly(database); con.row_factory=sqlite3.Row
    try:
        con.execute('BEGIN')
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require({'radius_moran_runs','radius_moran_points','radius_moran_results'}<=tables,
                'Нет радиальных запусков. Сначала MODE=calculate; старые local_moran_* здесь не используются.')
        row=(con.execute('SELECT * FROM radius_moran_runs ORDER BY run_id DESC LIMIT 1').fetchone()
             if run_id is None else con.execute('SELECT * FROM radius_moran_runs WHERE run_id=?',(run_id,)).fetchone())
        require(row is not None,'Радиальный запуск не найден.')
        run=dict(row); rid=run['run_id']
        points=[dict(r) for r in con.execute('SELECT * FROM radius_moran_points WHERE run_id=? ORDER BY point_id',(rid,))]
        results=[dict(r) for r in con.execute('SELECT * FROM radius_moran_results WHERE run_id=? ORDER BY point_id,metric',(rid,))]
        require(len(points)==run['n_points'] and len(results)==run['n_results'],'Неполный сохранённый запуск.')
        metrics=json.loads(run['config_json'])['metrics']
        validate_results([(p['point_id'],p['lon'],p['lat']) for p in points],metrics,results)
        return run,points,results
    finally:
        con.close()


def export_run(database,output,run_id):
    run,points,results=load_run(database,run_id)
    config=json.loads(run.pop('config_json')); source_meta=json.loads(run.pop('source_meta_json'))
    by_point={p['point_id']:{} for p in points}
    for result in results:
        prefix='normal' if result['metric']=='normal_mean_wm' else 'alongshore'
        for key in RESULT_FIELDS:
            if key not in {'point_id','metric'}:
                by_point[result['point_id']][f'{prefix}_{key}']=result[key]
    with TemporaryDirectory(prefix=f'.{output.stem}.',dir=output.parent) as directory:
        temporary=Path(directory)/'result.geojson'
        with temporary.open('w',encoding='utf-8') as stream:
            stream.write('{"type":"FeatureCollection","metadata":')
            json.dump(dict(export_created_at=datetime.now(UTC).isoformat(),run=run,parameters=config,source_meta=source_meta),
                      stream,ensure_ascii=False,allow_nan=False,indent=GEOJSON_INDENT)
            stream.write(',"features":[')
            for i,point in enumerate(points):
                if i: stream.write(',')
                pid=point['point_id']
                feature=dict(type='Feature',id=pid,geometry=dict(type='Point',coordinates=[point['lon'],point['lat']]),
                             properties=dict(point_id=pid,**by_point[pid]))
                json.dump(feature,stream,ensure_ascii=False,allow_nan=False,indent=GEOJSON_INDENT)
            stream.write(']}\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,output)
    logger.success('GeoJSON: радиальный run_id={}, features={}, {}',run['run_id'],len(points),output)


def main():
    logger.remove(); logger.add(sys.stderr,level=LOG_LEVEL,diagnose=False,backtrace=False)
    validate_settings()
    database,output=path_resolve(INDEX_DB),path_resolve(OUTPUT_GEOJSON)
    require(database.is_file(),f'Не найдена БД: {database}')
    calculate=MODE in {'calculate','calculate_export'}; export=MODE in {'export','calculate_export'}
    if export:
        output.parent.mkdir(parents=True,exist_ok=True); check_paths(database,output)
    started=time.monotonic()
    with ExitStack() as stack:
        locks={Path(f'{database}.lock')}
        if export: locks.add(Path(f'{output}.lock'))
        for lock in sorted(locks,key=str): stack.enter_context(file_lock(lock))
        if export:
            check_paths(database,output)
            require(not output.exists() or OVERWRITE_GEOJSON,f'GeoJSON уже существует: {output}. OVERWRITE_GEOJSON=True для замены.')
        selected=EXPORT_RUN_ID
        if calculate:
            watcher=readonly(database); stack.callback(watcher.close)
            expected,inode=version(watcher),identity(database)
            points,metrics,meta,digest=load_source(database)
            check=lambda:guard(watcher,expected,database,inode)
            check()
            logger.info('Радиальный Local Moran: точек={}, радиус={} м, метрики={}',len(points),RADIUS_M,metrics)
            logger.warning('Береговые части и барьеры не учитываются. Нормировка и фон перестановок — весь набор по метрике.')
            neighbors=radius_neighbors(points,RADIUS_M,MAX_DIRECTED_EDGES)
            results=[]
            for metric_no,metric in enumerate(metrics,1):
                results.extend(compute_metric(points,neighbors,metric,metric_no+2,metric_no))
            selected=publish(database,points,metrics,results,meta,digest,check)
            logger.success('Радиальный запуск {} сохранён: {} точек, {} результатов.',selected,len(points),len(results))
        if export:
            export_run(database,output,selected)
    logger.success('Завершено за {:.1f} с.',time.monotonic()-started)


if __name__=='__main__':
    main()