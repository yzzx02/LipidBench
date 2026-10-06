"""Shift actual mzML RT metadata and chromatogram time arrays, preserving spectral binaries."""
from pathlib import Path
import base64, hashlib, json, re, shutil, zlib
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET

from runtime import PROJECT_ROOT as ROOT, WORK
NS = {'m':'http://psi.hupo.org/ms/mzml'}
ET.register_namespace('',NS['m'])
ET.register_namespace('xsi','http://www.w3.org/2001/XMLSchema-instance')
SHIFTS = [-.5,-.2,-.1,0.,.1,.2,.5]

def scan_times(tree):
    return [p for p in tree.findall('.//m:spectrum//m:cvParam',NS) if p.get('accession')=='MS:1000016']

def time_binaries(tree):
    return [b for b in tree.findall('.//m:binaryDataArray',NS)
            if any(p.get('accession')=='MS:1000595' for p in b.findall('m:cvParam',NS))]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def non_time_binaries(root):
    h=hashlib.sha256(); n=0
    for b in root.findall('.//m:binaryDataArray',NS):
        codes={p.get('accession') for p in b.findall('m:cvParam',NS)}
        if 'MS:1000595' in codes: continue
        val=b.find('m:binary',NS)
        h.update((val.text or '').encode('ascii')); n+=1
    return h.hexdigest(),n

def decode_time(b):
    ps=b.findall('m:cvParam',NS); codes={p.get('accession') for p in ps}
    if 'MS:1000523' in codes: dtype=np.dtype('<f8')
    elif 'MS:1000521' in codes: dtype=np.dtype('<f4')
    else: raise ValueError('unsupported time array precision')
    if any(c.startswith('MS:10023') for c in codes): raise ValueError('unsupported Numpress time array')
    val=b.find('m:binary',NS); raw=base64.b64decode(val.text or '')
    compressed='MS:1000574' in codes
    if compressed: raw=zlib.decompress(raw)
    unit=next(p.get('unitAccession') for p in ps if p.get('accession')=='MS:1000595')
    factor=60. if unit=='UO:0000010' else 1. if unit=='UO:0000031' else None
    if factor is None: raise ValueError(unit)
    return np.frombuffer(raw,dtype=dtype).copy(),dtype,compressed,factor

def write_indexed(tree,dest):
    mzml=tree.find('m:mzML',NS)
    body=ET.tostring(mzml,encoding='UTF-8',xml_declaration=False)
    prefix=b'<?xml version="1.0" encoding="UTF-8"?>\n<indexedmzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.2_idx.xsd">\n'
    data=prefix+body+b'\n'; index_pos=len(data)
    indices=[]
    for kind in [b'spectrum',b'chromatogram']:
        offsets=[]
        for match in re.finditer(rb'<'+kind+rb'\s[^>]*\bid="([^"]+)"[^>]*>',body):
            offsets.append(b'<offset idRef="'+match.group(1)+b'">'+str(len(prefix)+match.start()).encode()+b'</offset>')
        if offsets: indices.append(b'<index name="'+kind+b'">\n'+b'\n'.join(offsets)+b'\n</index>')
    data+=b'<indexList count="'+str(len(indices)).encode()+b'">\n'+b'\n'.join(indices)+b'\n</indexList>\n'
    data+=b'<indexListOffset>'+str(index_pos).encode()+b'</indexListOffset>\n<fileChecksum>'
    checksum=hashlib.sha1(data).hexdigest().encode()
    data+=checksum+b'</fileChecksum>\n</indexedmzML>\n'
    dest.write_bytes(data)

def shift_one(source,dest,delta):
    tree=ET.parse(str(source)).getroot()
    original_hash,n=non_time_binaries(tree)
    times=scan_times(tree)
    vals=[]
    for p in times:
        unit=p.get('unitAccession'); factor=60. if unit=='UO:0000010' else 1. if unit=='UO:0000031' else None
        if factor is None: raise ValueError(unit)
        old=float(p.get('value')); new=old+delta*factor
        p.set('value',format(new,'.15g')); vals.append((old/factor,new/factor))
    ntime=0
    for b in time_binaries(tree):
        arr,dtype,compressed,factor=decode_time(b)
        shifted=(arr.astype(np.float64)+delta*factor).astype(dtype)
        raw=shifted.tobytes()
        if compressed: raw=zlib.compress(raw)
        encoded=base64.b64encode(raw).decode('ascii')
        b.find('m:binary',NS).text=encoded; b.set('encodedLength',str(len(encoded))); ntime+=1
    assert non_time_binaries(tree)==(original_hash,n)
    write_indexed(tree,dest)
    reread=ET.parse(str(dest)).getroot()
    assert non_time_binaries(reread)==(original_hash,n)
    newtimes=scan_times(reread)
    restored=[]
    for p in newtimes:
        factor=60 if p.get('unitAccession')=='UO:0000010' else 1
        restored.append(float(p.get('value'))/factor)
    assert np.allclose(np.array(restored),np.array(vals)[:,0]+delta,atol=1e-12,rtol=0)
    return dict(n_spectrum_rt=len(vals),n_chromatogram_time_arrays=ntime,n_preserved_binary_arrays=n,
                non_time_binary_sha256=original_hash,n_negative_scan_rt=sum(x<0 for x in restored),
                max_rt_shift_error_min=float(np.max(np.abs(np.array(restored)-np.array(vals)[:,0]-delta))))

def main():
    records=[]
    for row in pd.read_csv(WORK/'samples16.csv').itertuples(index=False):
        source=Path(row.source_path)
        for shift in SHIFTS:
            dest=WORK/'mzml'/f'shift_{shift:+.1f}'/source.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if shift==0:
                shutil.copyfile(source,dest)
                tree=ET.parse(str(source)).getroot(); h,n=non_time_binaries(tree)
                info=dict(n_spectrum_rt=len(scan_times(tree)),n_chromatogram_time_arrays=len(time_binaries(tree)),n_preserved_binary_arrays=n,non_time_binary_sha256=h,n_negative_scan_rt=0,max_rt_shift_error_min=0.)
            else: info=shift_one(source,dest,shift)
            records.append(dict(sample=row.sample,rt_shift=shift,mzml_path=str(dest),source_sha256=sha(source),shifted_sha256=sha(dest),**info))
        print('shifted mzML:',row.sample,flush=True)
    pd.DataFrame(records).to_csv(WORK/'shifted_mzml_manifest.csv',index=False)
    print('wrote',len(records),'mzML files; original binary intensities and m/z unchanged',flush=True)

if __name__=='__main__': main()
