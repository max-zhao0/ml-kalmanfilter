import tensorflow as tf
import tensorflow.keras as keras
import utils
import numpy
import sys

def create_hit_pairs(data, key):
    """
    Generate pairs of hits on adjacent layers and a corresponding score for if it is the correct propagation decision.

    Inputs
    --- 
    data        : hit data                                          : float (number of events, number of layers, max hits per layer, parameters per hit)
    key         : particle ids                                      : float (number of events, number of layers, max hits per layer)

    Outputs
    ---
    features    : Pairs of hits concatenated                        : float (number of training examples, parameters per hit * 2)
    labels      : Binary label for if pair is from the same track   : int   (number of training examples)
    """
    assert data.shape[:3] == key.shape

    features = []
    labels = []
    for event_no in range(data.shape[0]):
        for layer_no in range(data.shape[1] - 1):
            for hit_no in range(data.shape[2]):
                if key[event_no,layer_no,hit_no] == 0:
                    break
                
                # Search for matching hits on next layer
                found_match = False
                for hit_no2 in range(data.shape[2]):
                    if key[event_no,layer_no+1,hit_no2] == 0:
                        break
                    pair = tf.concat([data[event_no,layer_no,hit_no], data[event_no,layer_no+1,hit_no2]], 0)
                    if key[event_no,layer_no,hit_no] == key[event_no,layer_no+1,hit_no2]:
                        label = 1
                        found_match = True
                    else:
                        label = 0
                    features.append(pair)
                    labels.append(label)
                
                # If no matching hit is found on the next layer, the correct choice is to terminate the track, represented as all zeros hit
                terminal_pair = tf.concat([data[event_no,layer_no,hit_no], tf.zeros(data.shape[3], dtype=data.dtype)], 0)
                terminal_label = 0 if found_match else 1
                features.append(terminal_pair)
                labels.append(terminal_label)
    
    assert len(features) == len(labels)
    return tf.stack(features), tf.constant(labels)

def main(argv):
    hit_data_path = "smeared_hits.hdf5"
    model_path = "model_policy_nt"

    train_data, train_key, test_data, test_key, key_lst = utils.get_telescope_layer_data(hit_data_path, 0.8, 8, hit_info=["tx", "ty", "tz"])
    print(key_lst)
    print(train_data.shape)
    
    train_feat, train_labels = create_hit_pairs(train_data, train_key)
    test_feat, test_labels = create_hit_pairs(test_data, test_key)
    print(train_feat.shape)

    # Size can be modified for performance and accuracy needs
    PolicyModel = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    PolicyModel.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()]
    )
    for epochs in range(7):
        # Since training is not very intensive, evaluate and save after every epoch to check for overfitting
        
        PolicyModel.fit(train_feat, train_labels, epochs=1)
        print()
        result = PolicyModel.evaluate(test_feat, test_labels)
        print(result)
        PolicyModel.save(model_path+"_{:.4f}".format(result[0]))
        print()

    print("Finished!")

if __name__ == "__main__":
    main(sys.argv)
