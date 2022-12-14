#!/usr/bin/env python

import os,sys,math,glob,ROOT
import numpy as np
import h5py
from ROOT import gROOT, TFile, TTree


indata = "fatras/hits.root"
outdata = "hits.hdf5"
max_hits = 9*8 #maximum number of hits per event - this assumes 9 detector layers and 8 particles

def main(argv):
    gROOT.SetBatch(True)
    
    #define input file
    ntuple = TFile(indata)
    tree = ntuple.Get("hits")
    print("tree type:", type(tree))

    #find number of events
    nevents = 0
    for ientry,entry in enumerate(tree):
        if entry.event_id+1 > nevents:
            nevents = entry.event_id+1
    print("events:", nevents)

    #initialize arrays - fixed size for now, we need to change the way this information is stored later (this heavily depends on what kind of data structures we need as input to our ML algorithm later on -> for MCTS we might want to save it as something that looks more tree like)
    hit_layerid = np.zeros((nevents,max_hits))
    hit_x = np.zeros((nevents,max_hits))
    hit_y = np.zeros((nevents,max_hits))
    hit_z = np.zeros((nevents,max_hits))
    mom_x = np.zeros((nevents,max_hits))
    mom_y = np.zeros((nevents,max_hits))
    mom_z = np.zeros((nevents,max_hits))
    time = np.zeros((nevents,max_hits))
    ener = np.zeros((nevents,max_hits))
    part_id = np.zeros((nevents,max_hits))

    #process entries
    for ientry,entry in enumerate(tree):
        event_no = entry.event_id

        hit_no = 0
        while not hit_z[event_no,hit_no] == 0:
            hit_no += 1
        
        #store hit coordinates
        hit_layerid[event_no,hit_no] = entry.layer_id
        hit_x[event_no,hit_no] = entry.tx
        hit_y[event_no,hit_no] = entry.ty
        hit_z[event_no,hit_no] = entry.tz
        mom_x[event_no,hit_no] = entry.tpx
        mom_y[event_no,hit_no] = entry.tpy
        mom_z[event_no,hit_no] = entry.tpz
        time[event_no,hit_no] = entry.tt
        ener[event_no,hit_no] = entry.te
        part_id[event_no,hit_no] = entry.particle_id

    #create output file
    outfile = h5py.File(outdata, "w")
    grp_hitinfo = outfile.create_group("hitinfo")

    #save hit coordinates to output file
    grp_hitinfo.create_dataset("layer_id", data=hit_layerid)
    grp_hitinfo.create_dataset("tx", data=hit_x)
    grp_hitinfo.create_dataset("ty", data=hit_y)
    grp_hitinfo.create_dataset("tz", data=hit_z)
    grp_hitinfo.create_dataset("tpx", data=mom_x)
    grp_hitinfo.create_dataset("tpy", data=mom_y)
    grp_hitinfo.create_dataset("tpz", data=mom_z)
    grp_hitinfo.create_dataset("tt", data=time)
    grp_hitinfo.create_dataset("te", data=ener)
    grp_hitinfo.create_dataset("particle_id", data=part_id)

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
