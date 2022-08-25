import ROOT
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

hits = ROOT.RDataFrame("vol1", "fatras/measurements.root").AsNumpy()
#hits = ROOT.RDataFrame("track_finder_tracks", "reco/performance_track_finder.root").AsNumpy()
#hits = ROOT.RDataFrame("tracksummary", "reco/tracksummary_fitter.root").AsNumpy()

#for key in hits:
#	try:
#		print("{}\t\tmax: {:.2f}\tmin: {:.2f}\tlength: {}".format(key, max(hits[key]), min(hits[key]), len(hits[key])))
#	except:
#		print("{}:\t\t{}".format(key, hits[key]))

truth = ROOT.RDataFrame("hits", "fatras/hits.root").AsNumpy()
for i in range(15):
    for key in hits:
        print("{}:\t\t{}".format(key, hits[key][i]))
    print()
    for key in truth:
        print("{}:\t\t{}".format(key, truth[key][i]))
    print("\n\n\n")
