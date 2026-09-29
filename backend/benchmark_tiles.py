"""Reproducible bounded-read benchmark; virtual full-size files, no lunar downloads.

This measures allocations and read/seek volume, not physical disk throughput.
For real disk measurements, time backend.tile_cli against a locally held product. That route also streams SHA-256 over the full file.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
from time import perf_counter
import tracemalloc
from types import SimpleNamespace

from .lunar_core.io.ingestion import inspect_image, read_tile
from .lunar_core.io.metadata import ImageMetadata, RasterLayout, Tile


class VirtualRaster:
    def __init__(self, size):
        self.size=size
        self.reads=[]

    def stat(self):
        return SimpleNamespace(st_size=self.size)

    def open(self, mode):
        owner=self
        class Stream(io.RawIOBase):
            pos=0
            def seek(self, offset, whence=0):
                assert whence==0 and 0 <= offset <= owner.size
                self.pos=offset
            def read(self,count=-1):
                assert 0 <= count <= 2*1024*1024, 'unbounded read'
                assert self.pos+count <= owner.size, 'read past EOF'
                owner.reads.append((self.pos,count));self.pos+=count
                return b'\x01' * count
        return Stream()


def benchmark():
    results=[]
    for name,width,height,offset,dtype in [('LROC_M174353756RC_layout',5064,52224,5064,'<i2'),('TMC2_companion_layout',4000,148108,0,'<u2')]:
        layout=RasterLayout(width=width,height=height,offset=offset,dtype=dtype,sample_type='LSB_INTEGER' if dtype=='<i2' else 'UnsignedLSB2')
        file=VirtualRaster(layout.end_byte)
        record=ImageMetadata(source_format='PDS3' if offset else 'PDS4',label_location='attached' if offset else 'xml_sidecar',file_name=name+'.img',file_size=layout.end_byte,layout=layout)
        tile=Tile(x=width-512,y=height-512,width=512,height=512)
        tracemalloc.start();start=perf_counter()
        decoded=read_tile(file,record,tile)
        elapsed=perf_counter()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        expected=512*512*2
        assert decoded.bytes_read==expected==sum(n for _,n in file.reads)
        assert len(file.reads)==512 and peak < 64*1024**2
        assert max(pos+n for pos,n in file.reads)==layout.end_byte
        results.append({'layout':name,'full_file_bytes':layout.end_byte,'full_pixels':width*height,
            'tile':tile.model_dump(),'row_stride_bytes':layout.row_bytes,'pixel_bytes_read':decoded.bytes_read,
            'seek_count':len(file.reads),'peak_traced_allocation_bytes':peak,'elapsed_seconds':elapsed,
            'measurement':'virtual seekable file; excludes interpreter/native-library baseline and SHA-256 streaming pass'})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=benchmark()
    encoded=json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded,encoding='utf-8')
    print(encoded)


if __name__=='__main__':main()
