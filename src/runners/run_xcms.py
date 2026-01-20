import os
from pathlib import Path
from utils.config_io import load_config, get_base_dir

def run_xcms(input_dir,output_file,polarity,mz_tol,minwidth,maxwidth,noise=1000,sn=3,prefilter=3,):
    R_SCRIPT_PATH = os.path.dirname(__file__) + "/xcms.R"
    cmd=[
        "Rscript", str(R_SCRIPT_PATH),
        "--dir", str(input_dir),
        "--output", str(output_file),
        "polarity",str(polarity),
        "--mz_tol", str(mz_tol),
        "--minwidth", str(minwidth),
        "--maxwidth", str(maxwidth),
        "--noise", str(noise),
        "--sn", str(sn),
        "--prefilter", str(prefilter)
    ]
    try:
        os.system(" ".join(cmd))
    except Exception as e:
        print(f"An error occurred while running XCMS: {e}")
if __name__=="__main__":
    run_xcms()