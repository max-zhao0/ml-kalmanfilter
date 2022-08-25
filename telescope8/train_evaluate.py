import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sys
import h5py
import utils

def generate_random_tracks(data, key, fake_per_real=9):
    """
    Generates tracks to for training data. All real tracks are included along with a number of random tracks determined by fake_per_real.
    
    Input
    ---
    data        : hit data                      : float     (# of events, # of layers, max # of particles, # of hit parameters)
    key         : particle ids for each hit     : float64   (# of events, # of layers, max # of particles)
    
    Output
    ---
    features    : List of tensors for collections of tracks of different length from 1 to # of layers
    labels      : List of tensors for binary label indicating if each track in feature is a real or randomly generated one
    distr       : List of int for number of each tracks per each length
    """
    assert data.shape[:3] == key.shape
    tracks = [[] for _ in range(data.shape[1])]
    labels = [[] for _ in range(data.shape[1])]
    for event_no in range(data.shape[0]):
        
        # Find real tracks
        nrealtracks = 0
        for hit_no in range(data.shape[2]):
            if key[event_no,0,hit_no] != 0:
                nrealtracks += 1
                real_track = [data[event_no,0,hit_no]]
                track_id = key[event_no,0,hit_no]
                for layer_no in range(1,data.shape[1]):
                    found = False
                    for hit_no2 in range(data.shape[2]):
                        if key[event_no,layer_no,hit_no2] == track_id:
                            real_track.append(data[event_no,layer_no,hit_no2])
                            found = True
                            break
                    if not found:
                        break
                tracks[len(real_track)-1].append(tf.stack(real_track))
                labels[len(real_track)-1].append(1)
                
        # Generate fake tracks
        # May accidentally generate real tracks, but should have negligible effect on training as events become denser
        for _ in range(nrealtracks * fake_per_real):
            hit_nos = np.random.randint(0, data.shape[2], data.shape[1])
            first_hit_no = np.random.randint(0, nrealtracks)
            fake_track = [data[event_no,layer_no,first_hit_no]]
            for layer_no in range(1,data.shape[1]):
                if key[event_no,layer_no,hit_nos[layer_no]] != 0:
                    fake_track.append(data[event_no,layer_no,hit_nos[layer_no]])
                else:
                    break
            tracks[len(fake_track)-1].append(tf.stack(fake_track))
            labels[len(fake_track)-1].append(0)
      
    return [tf.stack(collec) for collec in tracks], [tf.constant(collec) for collec in labels], [len(collec) for collec in tracks]

def main(argv):
    hit_data_path = "smeared_hits.hdf5"
    model_path = "model_evaluate_nt"
    nparticles = 8

    train_data, train_key, test_data, test_key, key_lst = utils.get_telescope_layer_data(hit_data_path, 0.8, nparticles, hit_info=["tx", "ty", "tz"])
    print(train_data.shape)
    train_feat, train_labels, train_dist = generate_random_tracks(train_data, train_key)
    print(train_dist)
    print(train_feat[-1].shape)
    test_feat, test_labels, test_dist = generate_random_tracks(test_data, test_key)

    # Stacked LSTM network structure arbitrarily chosen currently
    Evaluation = keras.Sequential([
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16, return_sequences=True),
        keras.layers.LSTM(8),
        keras.layers.Dense(4, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    Evaluation.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()]
    )
    
    def training_generator(features, labels, dist, batch_size):
        # Generates training batches for LSTM input
        assert len(features) == len(labels) == len(dist)
        prob_nhits = np.array(dist) / np.sum(dist)
        shuffled_index = [tf.random.shuffle(tf.range(length)) for length in dist]
        shuffled_feat = [tf.gather(features[i], shuffled_index[i]) for i in range(len(features))]
        shuffled_lab = [tf.gather(labels[i], shuffled_index[i]) for i in range(len(labels))]
        curr_track = np.zeros(len(features))
        while True:
            # Ensures that each track in dataset is equally likely to be selexted
            nhit_index = np.random.choice(np.arange(len(features)), p=prob_nhits)
            batch_index = tf.cast(tf.range(curr_track[nhit_index], curr_track[nhit_index] + batch_size) % dist[nhit_index], tf.int32)
            batch_feat = tf.gather(features[nhit_index], batch_index)
            batch_lab = tf.gather(labels[nhit_index], batch_index)
            yield batch_feat, batch_lab
            # Stores current place in the dataset so that all tracks of a particlar size are fed before returning to the beginning
            curr_track[nhit_index] += batch_size

    Evaluation.fit(
        training_generator(train_feat, train_labels, train_dist, 32),
        steps_per_epoch=2150, # Number of tracks is approximately 2150 * 32
        epochs=2
    )
    # To do: build testing loop to evaluate model
    Evaluation.save(model_path)

    print("Finished")

if __name__ == "__main__":
    main(sys.argv)
