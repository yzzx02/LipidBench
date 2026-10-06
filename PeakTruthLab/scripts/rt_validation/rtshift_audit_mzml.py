from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
from rtshift_mzml import NS,scan_times,time_binaries,decode_time,non_time_binaries
from runtime import PROJECT_ROOT as ROOT, WORK as W, OUTPUT as O

def rt_minutes(tree):
    return np.array([float(p.get('value'))/(60 if p.get('unitAccession')=='UO:0000010' else 1) for p in scan_times(tree)])

def main():
    manifest=pd.read_csv(W/'shifted_mzml_manifest.csv');sources=pd.read_csv(W/'samples16.csv').set_index('sample')
    records=[]
    for sample,group in manifest.groupby('sample',sort=False):
        src=Path(sources.loc[sample,'source_path']);orig=ET.parse(src).getroot();times=rt_minutes(orig)
        h,n=non_time_binaries(orig);old_tic=[decode_time(b)[0]/decode_time(b)[3] for b in time_binaries(orig)]
        for row in group.itertuples(index=False):
            data=Path(row.mzml_path).read_bytes();tree=ET.fromstring(data)
            assert non_time_binaries(tree)==(h,n)
            rt=rt_minutes(tree);assert len(rt)==len(times)
            max_error=float(np.max(np.abs(rt-times-row.rt_shift)))
            dt_error=float(np.max(np.abs(np.diff(rt)-np.diff(times))))
            assert max_error<1e-10 and dt_error<1e-10
            tic=[decode_time(b)[0]/decode_time(b)[3] for b in time_binaries(tree)]
            assert len(tic)==len(old_tic)
            tic_error=max(float(np.max(np.abs(t-o-row.rt_shift))) for t,o in zip(tic,old_tic)) if tic else 0
            assert tic_error<1e-6
            index_offset=int(tree.find('m:indexListOffset',NS).text)
            assert data[index_offset:].startswith(b'<indexList ')
            for index in tree.findall('m:indexList/m:index',NS):
                for offset in index.findall('m:offset',NS):
                    pos=int(offset.text)
                    assert data.startswith(b'<'+index.get('name').encode()+b' ',pos)
            checksum=tree.find('m:fileChecksum',NS).text
            end=data.index(b'<fileChecksum>')+len(b'<fileChecksum>')
            assert hashlib.sha1(data[:end]).hexdigest()==checksum
            records.append(dict(sample=sample,rt_shift=row.rt_shift,n_scans=len(rt),n_non_time_binary_arrays=n,
                max_scan_rt_shift_error_min=max_error,max_scan_interval_error_min=dt_error,
                max_tic_rt_shift_error_min=tic_error,n_negative_scan_rt=int((rt<0).sum()),
                mz_intensity_binaries_unchanged=True,index_offsets_pass=True,sha1_checksum_pass=True))
        print('audit mzML',sample,flush=True)
    df=pd.DataFrame(records);df.to_csv(O/'mzml_shift_integrity_audit.csv',index=False)
    (O/'mzml_shift_integrity_audit.json').write_text(json.dumps(dict(n_files=len(df),all_checks_pass=True,
       spectral_mz_intensity_binaries='byte-for-byte unchanged',spectral_rt='all scan start times shifted, no clipping',
       chromatogram_times='TIC binary time array shifted; intensity array unchanged',
       negative_early_scans='retained; pyOpenMS and mzR/OnDisk reader smoke checks accepted them',
       max_scan_rt_shift_error_min=float(df.max_scan_rt_shift_error_min.max()),max_scan_interval_error_min=float(df.max_scan_interval_error_min.max())),indent=2),encoding='utf8')

if __name__=='__main__':main()
