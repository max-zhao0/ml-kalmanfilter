import tensorflow as tf
import tensorflow.keras as keras
import utils
import numpy
import sys
import multiprocessing as mp
import h5py
import time

def create_hit_pairs(data, key):
    """
    Generate pairs of hits on adjacent layers and a corresponding score for if it is the correct propagation decision.

    Inputs
    --- 
    data        : hit data                                          : float (number of events, number of layers, hits per layer + 1, parameters per hit)
    key         : particle ids                                      : float (number of events, number of layers, hits per layer + 1)

    Outputs
    ---
    features    : Pairs of hits concatenated                        : float (number of training examples, parameters per hit * 2)
    labels      : Binary label for if pair is from the same track   : int   (number of training examples)
    """ 
    assert data[0].shape[0] == data[-1].shape[0]
    global event_pair_gen
    def event_pair_gen(event_no):
        event_pairs = []
        event_labels = []

        for layer_no in range(data[0].shape[0] - 1):
            for hit_no in range(1, data[event_no,layer_no].shape[0]):
                # Search for matching hits on next layer
                found_match = False
                for hit_no2 in range(1, data[event_no,layer_no+1].shape[0]):
                    pair = tf.concat([data[event_no,layer_no,hit_no], data[event_no,layer_no+1,hit_no2]], 0)
                    if key[event_no,layer_no,hit_no] == key[event_no,layer_no+1,hit_no2]:
                        label = 1
                        found_match = True
                    else:
                        label = 0
                    event_pairs.append(pair)
                    event_labels.append(label)
                
                # If no matching hit is found on the next layer, the correct choice is to terminate the track, represented as all zeros hit
                terminal_pair = tf.concat([data[event_no,layer_no,hit_no], tf.zeros(data[event_no,layer_no,hit_no].shape[0], dtype=data.dtype)], 0)
                terminal_label = 0 if found_match else 1
                event_pairs.append(terminal_pair)
                event_labels.append(terminal_label)
        
        return event_pairs, event_labels

    with mp.Pool() as p:
        data_tuples = p.map(event_pair_gen, range(data.shape[0]), chunksize=1)
    p.join()
    pairs = []
    labels = []
    for event_no in range(len(data_tuples)):
        pairs += data_tuples[event_no][0]
        labels += data_tuples[event_no][1]

    del event_pair_gen

    assert len(pairs) == len(labels)
    return tf.stack(pairs), tf.constant(labels)

def main(argv):
    #assert len(argv) == 3
    #start_index = int(argv[1])
    #end_index = int(argv[2])
    hit_data_path = "/global/homes/m/max_zhao/mlkf/telescope32/smeared_hits.hdf5"
    train_data_path = "/global/homes/m/max_zhao/mlkf/telescope32/train_pairs.hdf5"
    test_data_path = "/global/homes/m/max_zhao/mlkf/telescope32/test_pairs.hdf5"
    nparticles = 32
    ntraining = 4000

    start_index = 4000
    end_index = 4250

    start = time.time()

    data, key = utils.get_telescope_layer_data(hit_data_path, nparticles, hit_info=["tx", "ty", "tz"])
    print(data.shape)

    test_pairs, test_labels = create_hit_pairs(data[ntraining:], key[ntraining:])
    print(test_pairs.shape)
    with h5py.File(test_data_path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("pairs", data=test_pairs)
        grp.create_dataset("labels", data=test_labels)
    print("Execution time:", time.time() - start)
    return 0

    train_pairs, train_labels = create_hit_pairs(data[start_index:end_index], key[start_index:end_index])
    print(train_pairs.shape)
    with h5py.File(train_data_path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("pairs", data=train_pairs)
        grp.create_dataset("labels", data=train_labels)

    end = time.time()
    print("Execution time:", end - start)

if __name__ == "__main__":
    main(sys.argv)
