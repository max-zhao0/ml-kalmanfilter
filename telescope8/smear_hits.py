import numpy as np
import h5py

def smear(indata, outdata):
    """
    Smear all hits from indata and saves to outdata
    
    Inputs
    ---
    indata      : string
    outdata     : string
    """
    in_file = h5py.File(indata, "r")
    hits = in_file["hitinfo"]

    nevents = len(hits[list(hits.keys())[0]])
    for key in hits.keys():
        assert len(hits[key]) == nevents
    print("Events:", nevents)
    
    # Standard deviation to smear each variable with
    scale = 0.5
    std = {
        "tx"  : 4*scale,
        "ty"  : 4*scale,
        "tz"  : 0,
        "tpx" : 0.2*scale,
        "tpy" : 0.2*scale,
        "tpz" : 0.1*scale,
        "te"  : 0.09*scale,
        "tt"  : 0.0118*scale
    }

    out_file = h5py.File(outdata, "w")
    grp_hitinfo = out_file.create_group("hitinfo")
    for key in hits.keys():
        if key in std:
            # Each variable is smeared independently with a normal distribution.
            grp_hitinfo.create_dataset(key, data=np.random.normal(hits[key], std[key]))
        else:
            grp_hitinfo.create_dataset(key, data=hits[key])
    
    in_file.close()
    out_file.close()

if __name__ == "__main__":
    smear("hits.hdf5", "smeared_hits.hdf5")
