import h5py
import tensorflow as tf
import numpy as np

def get_telescope_data(hit_data, training_prop):
    f = h5py.File(hit_data, "r")
    hits = f["hitinfo"]

    nevents = len(hits[list(hits.keys())[0]])
    for key in hits.keys():
        assert len(hits[key]) == nevents
    print("Events:", nevents)

    ntraining = int(training_prop * nevents)
    train_hits = {key:hits[key][:ntraining] for key in hits.keys()}
    test_hits = {key:hits[key][ntraining:] for key in hits.keys()}
    f.close()
    
    hit_info = ["tx", "ty", "tz", "tpx", "tpy", "tpz", "tt", "te"]
    train_lst = []
    test_lst = []
    key_lst = []
    for key in hit_info:
        train_lst.append(tf.transpose(tf.convert_to_tensor(train_hits[key])))
        test_lst.append(tf.transpose(tf.convert_to_tensor(test_hits[key])))
        key_lst.append(key)
    train_data = tf.transpose(tf.stack(train_lst))
    train_key = tf.convert_to_tensor(train_hits["particle_id"])
    test_data = tf.transpose(tf.stack(test_lst))
    test_key = tf.convert_to_tensor(test_hits["particle_id"])
    return train_data, train_key, test_data, test_key, key_lst

def create_hit_pairs(data, key):
    train = []
    labels = []
    for event_no in range(data.shape[0]):
        for hit_no in range(data.shape[1]):
            if key[event_no,hit_no] == 0:
                break
            for hit_no2 in range(hit_no+1, data.shape[1]):
                if key[event_no,hit_no2] == 0:
                    break
                train.append(tf.stack([data[event_no,hit_no], data[event_no,hit_no2]]))
                train.append(tf.stack([data[event_no,hit_no2], data[event_no,hit_no]]))
                if key[event_no,hit_no2] == key[event_no,hit_no]:
                    labels.append(1)
                    labels.append(1)
                else:
                    labels.append(0)
                    labels.append(0)
    assert len(train) == len(labels)
    return tf.stack(train), tf.constant(labels)

def unique_particles(data):
    hist = {}
    for event in data:
        num_unique = tf.unique(event)[0].shape[0] - 1
        if num_unique in hist:
            hist[num_unique] += 1
        else:
            hist[num_unique] = 1
    return hist

def get_telescope_layer_data(hit_data_path, training_prop, nparticles, nlayers=9, hit_info=["tx", "ty", "tz", "tt"], event_lim=None):
    """
    Reorganize hit data into layers for each events

    Inputs
    ---
    hit_data_path       : string
    training_prop       : float
    nparticles          : int
    nlayers             : int
    hit_info            : list
    event_lim           : int

    Outputs
    ---
    training_data       : float     (number of events, nlayers, nlayers * nparticles, parameters per hit)
    training_key        : float     (number of events, nlayers, nlayers * nparticles)
    testing_data        : float     (number of events, nlayers, nlayers * nparticles, parameters per hit)
    testing_key         : float     (number of events, mlayers, nlayers * nparticles)
    key_lst             : string    (parameters per hit)
    """
    f = h5py.File(hit_data_path, "r")
    hits = f["hitinfo"]

    nevents = len(hits[list(hits.keys())[0]])
    for key in hits.keys():
        assert len(hits[key]) == nevents
    print("Events:", nevents)
    
    if event_lim is not None:
        nevents = event_lim
        hits = {key:hits[key][:nevents] for key in hits}

    data = np.zeros((nevents, nlayers, nparticles, len(hit_info)))
    key = np.zeros((nevents, nlayers, nparticles))
    
    for event_no in range(nevents):
        for hit_no in range(nlayers * nparticles):
            layer_no = int(hits["layer_id"][event_no,hit_no] / 2 - 1)
            hit = np.array([hits[key][event_no,hit_no] for key in hit_info])
            for empty in range(nparticles+1):
                if empty >= nparticles:
                    raise Exception("Too many particles per layer")
                elif key[event_no,layer_no,empty] == 0:
                    data[event_no,layer_no,empty] = hit
                    key[event_no,layer_no,empty] = hits["particle_id"][event_no,hit_no]
                    break
    f.close()

    data = tf.convert_to_tensor(data)
    key = tf.convert_to_tensor(key)
    
    ntraining = int(training_prop * nevents)
    return data[:ntraining], key[:ntraining], data[ntraining:], key[ntraining:], hit_info

