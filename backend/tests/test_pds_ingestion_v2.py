"""Tiny synthetic fixtures: bounded reads, untrusted labels, independent standards."""
from __future__ import annotations
import hashlib
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lunar_core.io.metadata import HEADER_LIMIT, RasterLayout, Tile
from lunar_core.io.pds3_parser import parse_odl, parse_pds3, Quantity, Sentinel, image_offset
from lunar_core.io.pds4_raster import parse_pds4_raster
from lunar_core.io.ingestion import inspect_image, read_tile
from job_models import Settings
from main import create_app


def pds3(array, *, prefix=0, suffix=0, sample='LSB_INTEGER', extra='', special='NULL=-32768', record=1024):
    h, w = array.shape
    head = f'''PDS_VERSION_ID=PDS3
RECORD_TYPE=FIXED_LENGTH
RECORD_BYTES={record}
LABEL_RECORDS=1
^IMAGE=2
TARGET_NAME=MOON
PRODUCT_ID="synthetic"
{extra}
OBJECT=IMAGE
LINES={h}
LINE_SAMPLES={w}
SAMPLE_BITS={array.dtype.itemsize * 8}
SAMPLE_TYPE={sample}
LINE_PREFIX_BYTES={prefix}
LINE_SUFFIX_BYTES={suffix}
{special}
END_OBJECT=IMAGE
END
'''.encode()
    assert len(head) <= record
    return head.ljust(record, b' ') + b''.join(b'p' * prefix + row.tobytes() + b's' * suffix for row in array)


def pds4(name, width, height, sample='UnsignedLSB2', *, offset=0, extra='', specials=''):
    return f'''<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
<Identification_Area><logical_identifier>urn:test:synthetic</logical_identifier></Identification_Area>
{extra}<File_Area_Observational><File><file_name>{name}</file_name></File>
<Array_2D_Image><offset unit="byte">{offset}</offset><axes>2</axes><axis_index_order>Last Index Fastest</axis_index_order>
<Element_Array><data_type>{sample}</data_type><scaling_factor>0.5</scaling_factor><value_offset>1</value_offset></Element_Array>
<Axis_Array><axis_name>Line</axis_name><elements>{height}</elements><sequence_number>1</sequence_number></Axis_Array>
<Axis_Array><axis_name>Sample</axis_name><elements>{width}</elements><sequence_number>2</sequence_number></Axis_Array>
{specials}</Array_2D_Image></File_Area_Observational></Product_Observational>'''.encode()


def test_odl_scopes_comments_sequences_units_and_end():
    data = b'''/* END */ pds_version_id=PDS3\r\n
NOTE="continued\n END\n text" A=16#-FF# B=0.414400 <ms> C=(0,16,69) D=UNK
OBJECT=IMAGE NULL=-32768 GROUP=INNER KEY=1 END_GROUP=INNER END_OBJECT=IMAGE
OBJECT=OTHER NULL=NULL END_OBJECT
LRO:EXPOSURE=1.2e-3 END
'''
    root, end = parse_odl(data)
    assert root.get('A') == -255 and root.get('B') == Quantity(.4144, 'MS')
    assert root.get('C') == (0, 16, 69) and root.get('D') == Sentinel('UNK')
    assert root.children[0].get('NULL') == -32768
    assert root.children[1].get('NULL') == Sentinel('NULL')
    assert root.children[0].children[0].get('KEY') == 1
    assert data[end:].strip() == b''


@pytest.mark.parametrize('data', [b'OBJECT=A END_GROUP=A END', b'OBJECT=A END', b'A={1,2} END', b'A=1e999 END', b'^STRUCTURE="../bad" END', b'A="fake \\" END', b'A=(1,2 END'])
def test_odl_rejects_unsupported_or_malformed(data):
    with pytest.raises(ValueError):
        parse_odl(data)


def test_odl_header_and_depth_limits_and_fuzz():
    with pytest.raises(ValueError):
        parse_odl(b'A="' + b'x' * HEADER_LIMIT + b'" END')
    with pytest.raises(ValueError):
        parse_odl((b'OBJECT=A ' * 40) + b'END')
    rng = random.Random(91)
    for _ in range(400):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 1024)))
        try:
            root, end = parse_odl(data)
            assert 0 < end <= len(data)
        except ValueError:
            pass


def test_one_based_pointers_and_detached_containment():
    assert image_offset(2, 5064) == 5064
    assert image_offset(Quantity(5065, 'BYTES'), None) == 5064
    for p in [0, -1, '../x.img', '/tmp/x', 'https://example.com/x', ('x.img', 1), 2**63]:
        with pytest.raises(ValueError):
            image_offset(p, 5064)


def test_signed_rows_padding_scaling_mask_and_window(tmp_path):
    array = np.array([[-32768, 0, -9, 12], [14, 18, 22, 32767], [3, 7, 11, 13]], dtype='<i2')
    data = pds3(array, prefix=3, suffix=5, special='NULL=-32768\nHIGH_REPR_SATURATION=32767\nSCALING_FACTOR=0.5\nOFFSET=2')
    path = tmp_path / 'test.IMG'; path.write_bytes(data)
    meta = inspect_image(path, path.name)
    assert meta.layout.dtype == '<i2' and meta.layout.row_bytes == 16
    tile = read_tile(path, meta, Tile(x=1, y=0, width=3, height=2))
    np.testing.assert_array_equal(tile.raw, array[:2, 1:])
    assert tile.valid[0, 0] and tile.calibrated[0, 0] == 2
    assert not tile.valid[1, 2] and np.isnan(tile.calibrated[1, 2])
    assert tile.bytes_read == 12
    assert meta.gsd_meters is None and meta.emission_angle_deg is None
    assert meta.sha256 == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('dtype,sample', [('>i2','MSB_INTEGER'), ('>u2','MSB_UNSIGNED_INTEGER'), ('<u2','LSB_UNSIGNED_INTEGER'), ('>f4','IEEE_REAL'), ('<f4','PC_REAL')])
def test_endian_types(tmp_path, dtype, sample):
    array = np.array([[0, 7], [123, 42]], dtype=dtype)
    path = tmp_path / 'a.img'; path.write_bytes(pds3(array, sample=sample, special=''))
    tile = read_tile(path, inspect_image(path, path.name), Tile(x=0,y=0,width=2,height=2))
    np.testing.assert_array_equal(tile.raw, array)
    assert tile.valid.all()


def test_unsigned_pds4_special_and_float_nonfinite(tmp_path):
    path = tmp_path / 'a.img'
    array = np.array([[0, 65535], [32768, 12]], dtype='<u2'); path.write_bytes(array.tobytes())
    xml = pds4('a.img',2,2,specials='<Special_Constants><missing_constant>65535</missing_constant></Special_Constants>')
    meta = inspect_image(path, path.name, xml=xml)
    tile = read_tile(path,meta,Tile(x=0,y=0,width=2,height=2))
    assert tile.raw[1,0] == 32768 and tile.valid[0,0] and not tile.valid[0,1]
    assert tile.calibrated[1,0] == 16385
    array = np.array([[0, np.nan], [np.inf, 12]], dtype='<f4'); path.write_bytes(pds3(array,sample='PC_REAL',special=''))
    tile = read_tile(path,inspect_image(path,path.name),Tile(x=0,y=0,width=2,height=2))
    np.testing.assert_array_equal(tile.valid, [[True,False],[False,True]])


def test_public_layout_record_not_row():
    header = (Path(__file__).parent/'fixtures'/'public_lroc_layout.odl').read_bytes()
    meta = parse_pds3(header,'M174353756RC.IMG',5064 + 52224 * 10128)
    assert meta.layout.offset == 5064 and meta.layout.width == 5064
    assert meta.layout.height == 52224 and meta.layout.row_bytes == 10128
    assert meta.layout.dtype == '<i2' and meta.layout.special_constants == [-32768,-32767,-32766,-32764,-32765]


def test_truncation_overflow_and_bad_sidecar(tmp_path):
    path = tmp_path/'a.img'; data = pds3(np.zeros((2,2),dtype='<i2')); path.write_bytes(data[:-1])
    with pytest.raises(ValueError,match='Truncated'):
        inspect_image(path,path.name)
    with pytest.raises(ValueError):
        RasterLayout(width=2**62,height=4,dtype='<i2',sample_type='LSB_INTEGER')
    path.write_bytes(data)
    with pytest.raises(ValueError,match='Supplied XML is invalid'):
        inspect_image(path,path.name,xml=b'<bad>')
    forced = inspect_image(path,path.name,'pds3',xml=b'<bad>')
    assert any('Malformed' in w for w in forced.warnings)
    with pytest.raises(ValueError,match='file_name'):
        inspect_image(path,path.name,xml=pds4('different.img',2,2))


def test_xml_entities_and_paths():
    for xml in (b'<!DOCTYPE x [<!ENTITY secret SYSTEM "file:///secret">]><x/>', pds4('../x',2,2)):
        with pytest.raises(ValueError):
            parse_pds4_raster(xml,'a.img',8)


@pytest.fixture
def api(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, 'enqueue', lambda *args: None)
    settings = Settings(storage_dir=tmp_path/'jobs', always_eager=True, max_active_jobs=20)
    with TestClient(create_app(settings)) as client:
        yield client, settings


@pytest.mark.parametrize('standards', [('PDS4','PDS4'),('PDS3','PDS4'),('PDS4','PDS3'),('PDS3','PDS3'),('IMAGE_ONLY','IMAGE_ONLY'),('PDS3','IMAGE_ONLY')])
def test_mixed_api_pair_modes_and_provenance(api, standards):
    from job_store import JobStore
    client, settings = api
    files = {}
    for side, standard in zip(('source','reference'),standards):
        a = np.arange(64*64,dtype='<i2').reshape(64,64)
        if standard == 'PDS3':
            files[side+'_file']=(side+'.img',pds3(a))
        elif standard == 'PDS4':
            files[side+'_file']=(side+'.img',a.astype('<u2').tobytes())
            files[side+'_label']=(side+'.xml',pds4(side+'.img',64,64))
        else:
            files[side+'_file']=(side+'.png',cv2.imencode('.png',a.astype('uint16'))[1].tobytes())
    params={'schema_version':2,'source_tile':{'x':0,'y':0,'width':64,'height':64},'reference_tile':{'x':0,'y':0,'width':64,'height':64},'confirm_unknown_overlap':True}
    response=client.post('/api/v1/registration/jobs',files=files,data={'params':json.dumps(params)})
    assert response.status_code == 202,response.text
    record=JobStore(settings.storage_dir).get(response.json()['job_id'],include_result=True)['params']
    assert [record[s+'_record']['source_format'] for s in ('source','reference')] == list(standards)
    assert all(record[s+'_record']['sha256'] for s in ('source','reference'))
    assert record['overlap_status']=='unknown'


def test_mixed_known_overlap_e2e(tmp_path):
    import csv
    from rasterio.io import MemoryFile
    # Known-overlap synthetic texture; no assumption about real mission overlap.
    rng = np.random.default_rng(772)
    image = np.full((608,608),64,dtype=np.uint8)
    for cy in range(56,568,64):
        for cx in range(56,568,64):
            image[cy-14:cy+14,cx-14:cx+14] = cv2.GaussianBlur(rng.integers(0,256,(28,28),dtype=np.uint8),(3,3),.6)
    source = (image[:600,:600].astype('<i2')*100)
    reference = (image[4:604,6:606].astype('<u2')*100)
    source[:35,:] = -32768
    tile={'x':24,'y':24,'width':512,'height':512}
    files={'source_file':('source.img',pds3(source)), 'reference_file':('reference.img',reference.tobytes()),
           'reference_label':('reference.xml',pds4('reference.img',600,600))}
    settings=Settings(storage_dir=tmp_path/'jobs',always_eager=True)
    with TestClient(create_app(settings)) as client:
        response=client.post('/api/v1/registration/jobs',files=files,data={'params':json.dumps({
            'schema_version':2,'source_tile':tile,'reference_tile':tile,'confirm_unknown_overlap':True})})
        assert response.status_code==202,response.text
        result=client.get(f"/api/v1/registration/jobs/{response.json()['job_id']}/results").json()
        assert result['status']=='review_required',result
        assert result['matching_mode']=='image_only' and result['overlap_status']=='unknown'
        assert result['routing']['rationales'][0]=='insufficient telemetry'
        assert result['metrics']['rmse_basis']=='withheld_feature_correspondences',result
        assert result['metrics']['rmse_px'] < .5,result
        assert not result['artifacts']['georeferenced']
        assert result['metadata']['source']['source_format']=='PDS3'
        assert result['metadata']['reference']['source_format']=='PDS4'
        tiff=client.get(result['artifacts']['geotiff_download_url']).content
        with MemoryFile(tiff) as memory, memory.open() as ds:
            assert ds.crs is None and ds.shape==(512,512) and ds.dtypes==('float64',)
            assert (ds.dataset_mask()==0).any() and (ds.dataset_mask()==255).any()
        rows=list(csv.DictReader(io.StringIO(client.get(result['artifacts']['tie_points_csv_url']).text)))
        assert rows and all(r['lat']==r['lon']=='' for r in rows)
        assert all(abs(float(r['src_full_x'])-float(r['src_x'])-24)<1e-6 for r in rows)
        errors=[[float(r['ref_x'])-float(r['src_x'])+6,float(r['ref_y'])-float(r['src_y'])+4] for r in rows]
        assert np.sqrt(np.mean(np.square(errors))) < .5
        dossier=client.get(result['artifacts']['dossier_url']).json()
        assert dossier['metadata']['source']['product_id']=='synthetic'
        assert dossier['export']['independent_ground_validation'] is False
        assert not list(settings.storage_dir.glob('job_lunar_*/*.img'))


@pytest.mark.parametrize('standard',['PDS3','PDS4'])
def test_local_tile_manifest_roundtrip(tmp_path,standard):
    # Import package entry point using project root; production CLI is python -m backend.tile_cli.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from backend.tile_cli import extract
    image=tmp_path/'large.img'; xml=None
    array=np.arange(80*96,dtype='<i2').reshape(80,96)
    if standard=='PDS3':
        image.write_bytes(pds3(array))
    else:
        image.write_bytes(array.astype('<u2').tobytes())
        xml=tmp_path/'large.xml';xml.write_bytes(pds4('large.img',96,80))
    output=tmp_path/'tile.tif'
    report=extract(image,output,Tile(x=7,y=11,width=40,height=32),xml)
    manifest=Path(report['provenance']).read_bytes()
    meta=inspect_image(output,output.name,provenance=manifest)
    assert meta.source_format==standard and meta.lineage['full_image_origin']==[7,11]
    assert meta.ingestion_route=='local_tile' and meta.geometry_status=='unknown'
    with pytest.raises(ValueError, match='Original-file checksum'):
        inspect_image(output, output.name, provenance=manifest, verify_checksum=True)
    tile=read_tile(output,meta,Tile(x=0,y=0,width=40,height=32))
    np.testing.assert_array_equal(tile.raw,array[11:43,7:47])
    bad=json.loads(manifest);bad['tile_sha256']='0'*64
    with pytest.raises(ValueError,match='SHA-256'):
        inspect_image(output,output.name,provenance=json.dumps(bad).encode())


def test_lroc_profile_and_checksum_scope(tmp_path):
    import hashlib
    array=np.arange(64*64,dtype='<i2').reshape(64,64)
    extra='''DATA_SET_ID="LRO-L-LROC-3-CDR-V1.1"
PRODUCT_TYPE=CDR
FRAME_ID=RIGHT
START_TIME=2011-01-01T00:00:00
STOP_TIME=2011-01-01T00:01:00
'''
    data=pds3(array,extra=extra,special='NULL=-32768\nMD5_CHECKSUM="'+hashlib.md5(array.tobytes()).hexdigest()+'"')
    data=data[:1024].rstrip(b' ').replace(b'PRODUCT_ID="synthetic"',b'PRODUCT_ID=M174353756RC').ljust(1024,b' ')+data[1024:]
    # Both product declarations have equal length so the raster pointer remains unchanged.
    assert data[1024:]==array.tobytes()
    path=tmp_path/'M174353756RC.IMG';path.write_bytes(data)
    result=inspect_image(path,path.name,verify_checksum=True)
    assert result.checksum['scope']=='image_object' and result.checksum['status']=='verified_transport_only'
    assert result.emission_angle_deg is None
    mixed_times = data[:1024].rstrip(b' ').replace(b'STOP_TIME=2011-01-01T00:01:00', b'STOP_TIME=2011-01-01T00:01:00Z').ljust(1024, b' ') + data[1024:]
    with pytest.raises(ValueError, match='time interval'):
        parse_pds3(mixed_times, path.name, len(mixed_times))
    for filename in ('wrong.IMG','../M174353756RC.IMG'):
        with pytest.raises(ValueError):inspect_image(path,filename)
    path.write_bytes(data[:-1]+b'x')
    with pytest.raises(ValueError,match='checksum mismatch'):
        inspect_image(path,path.name,verify_checksum=True)


def test_byte_pointer_repeated_keys_and_layout_rejections():
    data=pds3(np.zeros((2,2),dtype='<i2'))
    header=data[:1024].replace(b'^IMAGE=2',b'^IMAGE=1025 <BYTES>')
    assert parse_pds3(header,'a.img',len(data)).layout.offset==1024
    for replacement in [b'LINES=9223372036854775807', b'LINES=2\nLINES=3', b'LINES=2\nBANDS=2', b'LINES=2\nENCODING_TYPE=COMPRESSED']:
        with pytest.raises(ValueError):parse_pds3(data[:1024].replace(b'LINES=2',replacement),'a.img',len(data))
    with pytest.raises(ValueError):parse_odl(b'PDS_VERSION_ID=PDS3 END=4')
    assert parse_pds3(data[:1024].replace(b'PDS_VERSION_ID=PDS3',b'pds_version_id /* hi */ = pds3'),'a.img',len(data)).source_format=='PDS3'


def test_grid_overlap_wrapping_and_no_bbox_grid(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from lunar_core.io.geotiff_exporter import MOON_2000_WKT
    from lunar_core.io.ingestion import overlap_for
    records=[]
    for name,west in [('a.tif',359.8),('b.tif',-.1),('c.tif',60)]:
        path=tmp_path/name
        with rasterio.open(path,'w',driver='GTiff',width=64,height=64,count=1,dtype='uint16',crs=MOON_2000_WKT,transform=from_origin(west,10,.01,.01)) as ds:
            ds.write(np.ones((64,64),np.uint16),1)
        records.append(inspect_image(path,name))
    tile=Tile(x=0,y=0,width=64,height=64)
    assert overlap_for(records[0],records[1],tile,tile)=='verified_grid_overlap'
    assert overlap_for(records[0],records[2],tile,tile)=='nonoverlap'
    assert records[0].grid and records[0].footprint
    xml=pds4('x.img',2,2,extra='<west_bounding_coordinate>350</west_bounding_coordinate><east_bounding_coordinate>10</east_bounding_coordinate><north_bounding_coordinate>5</north_bounding_coordinate><south_bounding_coordinate>0</south_bounding_coordinate>')
    assert parse_pds4_raster(xml,'x.img',8).grid is None


@pytest.mark.parametrize('mode,xml,status', [('pds4',None,422),('none',None,422),('auto',b'<bad>',422),('pds3',b'<bad>',200)])
def test_api_inspection_forced_modes(api,mode,xml,status):
    client,_=api
    files={'file':('a.img',pds3(np.ones((64,64),dtype='<i2')))}
    if xml:files['label']=('bad.xml',xml)
    response=client.post('/api/v1/registration/inspect',files=files,data={'mode':mode})
    assert response.status_code==status,response.text
    if status==200:
        assert response.json()['state']=='PDS3 attached label'
        assert any('XML' in w for w in response.json()['metadata']['warnings'])


def test_quota_stale_unknown_and_cancellation_cleanup(api,tmp_path):
    from dataclasses import replace
    from job_store import JobStore
    import tasks
    client,settings=api
    files={s+'_file':(s+'.img',pds3(np.ones((64,64),dtype='<i2'))) for s in ('source','reference')}
    tile={'x':0,'y':0,'width':64,'height':64}
    params={'schema_version':2,'source_tile':tile,'reference_tile':tile,'confirm_unknown_overlap':True}
    for changes in ({'source_sha256':'0'*64},{'confirm_unknown_overlap':False},{'source_tile':{**tile,'x':1000}}):
        response=client.post('/api/v1/registration/jobs',files=files,data={'params':json.dumps({**params,**changes})})
        assert response.status_code==422,response.text
    assert not list(settings.storage_dir.glob('.upload-*'))
    response=client.post('/api/v1/registration/jobs',files=files,data={'params':json.dumps(params)})
    job=response.json()['job_id']
    # Cancellation remains possible even when no further upload can be admitted.
    with TestClient(create_app(replace(settings, max_storage_bytes=1))) as full:
        assert full.post(f'/api/v1/registration/jobs/{job}/cancel').status_code==200
    assert not JobStore(settings.storage_dir).directory(job).exists()
    assert JobStore(settings.storage_dir).get(job)['status']=='failed'
    tasks.process_registration(job,str(settings.storage_dir)) # cancelled job cannot be claimed later
    for overrides,expected in [({'max_upload_bytes':10},413),({'max_memory_bytes':1024},422),({'max_storage_bytes':1},429),({'max_tile_pixels':100},422)]:
        with TestClient(create_app(replace(settings,**overrides))) as limited:
            response=limited.post('/api/v1/registration/jobs',files=files,data={'params':json.dumps(params)})
            assert response.status_code==expected,response.text
    assert not list(settings.storage_dir.glob('.upload-*'))
    with JobStore(settings.storage_dir).connect() as db:
        assert db.execute('select count(*) from reservations').fetchone()[0]==0


def test_pds4_standard_target_identification():
    for name in ('Moon', 'Mars'):
        xml = pds4('x.img', 2, 2, extra=f'<Observation_Area><Target_Identification><name>{name}</name></Target_Identification></Observation_Area>')
        if name == 'Mars':
            with pytest.raises(ValueError, match='Moon target'):
                parse_pds4_raster(xml, 'x.img', 8)
        else:
            assert parse_pds4_raster(xml, 'x.img', 8).target == 'moon'


def test_no_invalid_features_or_entropy_from_valid_zero():
    from lunar_core.matching.matcher import SIFTMatcher
    from lunar_core.matching.router import compute_illumination_entropy
    from lunar_core.preprocess.wallis import apply_wallis_filter
    image=np.zeros((128,128),np.uint8)
    image[40:90,40:90]=255
    mask=np.ones_like(image,dtype=bool);mask[35:95,35:95]=False
    assert compute_illumination_entropy(image,mask)==0
    values=np.array([[0,0,255,255]],dtype=np.uint8)
    assert compute_illumination_entropy(values,np.ones_like(values,dtype=bool))==1
    conditioned=apply_wallis_filter(image,mask=mask)
    assert (conditioned[~mask]==0).all()
    assert len(SIFTMatcher().match(conditioned,conditioned,mask_src=mask,mask_ref=mask))==0


def test_georeferenced_export_requires_reference_header_grid(tmp_path):
    import csv
    import rasterio
    from rasterio.transform import from_origin
    from lunar_core.io.geotiff_exporter import MOON_2000_WKT
    from lunar_core.io.window_exporter import write_bundle
    from job_models import ValidatedJobParams
    source=tmp_path/'source.img';source.write_bytes(pds3(np.arange(64*64,dtype='<i2').reshape(64,64)))
    reference=tmp_path/'reference.tif'
    with rasterio.open(reference,'w',driver='GTiff',width=64,height=64,count=1,dtype='uint16',crs=MOON_2000_WKT,transform=from_origin(359.9,12,.001,.001)) as ds:ds.write(np.ones((64,64),np.uint16),1)
    src,ref=inspect_image(source,source.name),inspect_image(reference,reference.name)
    tile=Tile(x=8,y=10,width=32,height=32)
    params=ValidatedJobParams(source_record=src,reference_record=ref,source_tile=tile,reference_tile=tile)
    result={'metadata':{'source':src.model_dump(),'reference':ref.model_dump()}}
    valid=np.ones((32,32),bool);valid[0,:]=False
    write_bundle(tmp_path/'out',np.zeros((32,32),np.float64),valid,result,params,np.array([[0,1,2,3.]]),np.array([.1]),np.array([.9]),['LG-0001'])
    with rasterio.open(tmp_path/'out'/'registered_output.tif') as ds:
        assert ds.crs is not None and ds.dataset_mask()[1,1]==255 and ds.read(1)[1,1]==0
        assert abs(ds.transform.c-(-.092))<1e-9 and abs(ds.transform.f-11.99)<1e-9
    rows=list(csv.DictReader((tmp_path/'out'/'tiepoints.csv').open()))
    assert abs(float(rows[0]['lat'])-(12-(10+3.5)*.001))<1e-10
    assert rows[0]['src_full_x']=='8.0' and result['export']['independent_ground_validation'] is False
