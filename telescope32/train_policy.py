import tensorflow as tf
import h5py
import sys

def main(argv):
    train_data_path = "/global/homes/m/max_zhao/mlkf/telescope32/test_pairs.hdf5"
    test_data_path = ""
    
    f = h5py.File(train_data_path, "r")
    train_feat = tf.convert_to_tensor(f["data"]["pairs"])
    train_labels = tf.convert_to_tensor(f["data"]["labels"])
    f.close()

    print(train_feat)
    print(train_labels)
    return 0

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


if __name__ == "__main__":
    main(sys.argv)
