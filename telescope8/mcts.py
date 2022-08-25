import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
import sys
from scipy import stats

def get_telescope_layer_data(hit_data, training_prop, nparticles, nlayers=9, hit_info=["tx", "ty", "tz", "tt"], event_lim=None):
    f = h5py.File(hit_data, "r")
    hits = f["hitinfo"]

    nevents = len(hits[list(hits.keys())[0]])
    for key in hits.keys():
        assert len(hits[key]) == nevents
    print("Events:", nevents)
    
    if event_lim is not None:
        nevents = event_lim
        hits = {key:hits[key][:nevents] for key in hits}
    
    data = []
    key = []
    for event_no in range(nevents):
        event_data = [[[0]*len(hit_info)] for _ in range(nlayers)]
        event_key = [[0] for _ in range(nlayers)]
        for hit_no in range(nlayers * nparticles):
            if hits["particle_id"][event_no,hit_no] != 0:
                layer_no = int(hits["layer_id"][event_no,hit_no] / 2 - 1)
                hit = [hits[key][event_no,hit_no] for key in hit_info]
                event_data[layer_no].append(hit)
                event_key[layer_no].append(hits["particle_id"][event_no,hit_no])
        data.append(tf.ragged.constant(event_data, dtype=tf.float32))
        key.append(tf.ragged.constant(event_key, dtype=tf.float64))
    f.close()

    data = tf.ragged.stack(data)
    key = tf.ragged.stack(key)
    
    ntraining = int(training_prop * nevents)
    return data[:ntraining], key[:ntraining], data[ntraining:], key[ntraining:], hit_info

def naive_tracking(event, Policy, truth=None):
    if truth is None:
        truth = tf.ragged.map_flat_values(lambda x: 0, event)
    
    tracks = []
    truth_tracks = []
    for track_no in range(1, event[0].shape[0]):
        track = [event[0,track_no]]
        truth_track = [truth[0,track_no]]

        for layer_no in range(1, event.shape[0]):
            current_hit_expanded = tf.repeat(tf.expand_dims(track[-1], 0), event[layer_no].shape[0], 0) 
            policy_input = tf.concat([current_hit_expanded, event[layer_no].to_tensor()], 1)
            distr = tf.reshape(Policy(policy_input), [-1])
            choice = np.argmax(distr)
            if choice == 0:
                break
            else:
                track.append(event[layer_no,choice])
                truth_track.append(truth[layer_no,choice])

        tracks.append(track)
        truth_tracks.append(truth_track)
    
    return tracks if truth is None else (tracks, truth_tracks)

class Tree:
    def __init__(self, parent, coordinate, hit, truth, branches=[]):
        self.parent = parent
        self.hit = hit
        self.truth = truth
        self.branches = branches.copy()
        self.coord = coordinate

class Edge:
    u_bias = 1
    def __init__(self, top, prior, bottom=None):
        self.top = top
        self.bottom = bottom
        self.prior = prior
        self.count = 0
        self.result_sum = 0
    def action_val(self):
        return self.result_sum / self.count if self.count else 0.5
    def sim_val(self):
        return self.action_val() + self.u_bias * self.prior / (1 + self.count)
    def update(self, result):
        self.count += 1
        self.result_sum += result

def tracking(event, Policy, Evaluate, truth=None):
    if truth is None:
        truth = tf.ragged.map_flat_values(lambda x: 0, event)
    reduce = lambda v: np.array(v) / tf.reduce_sum(v)

    tracks = []
    truth_tracks = []
    for track_no in range(1, event[0].shape[0]):
        track = [event[0,track_no]]
        truth_track = [truth[0,track_no]]
        hit_tree = Tree(None, (0, track_no), event[0,track_no], truth[0,track_no])

        for layer_no in range(event.shape[0] - 1): # No need to iterate on final layer
            niterations = 10 - layer_no # Can replace with better heuristic
            
            for _ in range(niterations):
                candidate = track.copy()

                # Selection
                curr_tree = hit_tree
                while curr_tree.branches:
                    sim_choice = np.argmax([branch.sim_val() for branch in curr_tree.branches])
                    curr_tree = curr_tree.branches[sim_choice].bottom
                    candidate.append(curr_tree.hit)
                
                first_choice = None
                if curr_tree.hit is not None:
                    # Expansion
                    if curr_tree.coord[0] >= event.shape[0] - 1: # End of detector
                        final_edge = Edge(curr_tree, 1)
                        final_edge.bottom = Tree(final_edge, None, None, None)
                        curr_tree.branches.append(final_edge)
                    else:
                        next_layer = curr_tree.coord[0] + 1
                        next_layer_hits = event[next_layer].to_tensor() # all hits on next layer
                        curr_hit_expanded = tf.repeat(tf.expand_dims(curr_tree.hit, 0), next_layer_hits.shape[0], 0)
                        priors_input = tf.concat([curr_hit_expanded, next_layer_hits], 1)
                        priors = tf.reshape(Policy(priors_input), [-1])
                        end_edge = Edge(curr_tree, priors[0].numpy())
                        end_edge.bottom = Tree(end_edge, None, None, None)
                        curr_tree.branches.append(end_edge)
                        for i in range(1, priors.shape[0]):
                            new_edge = Edge(curr_tree, priors[i].numpy())
                            new_edge.bottom = Tree(new_edge, (next_layer, i), next_layer_hits[i], truth[next_layer, i])
                            curr_tree.branches.append(new_edge)

                    # MC Playout
                    first_choice = np.random.choice(len(curr_tree.branches), p=reduce([branch.prior for branch in curr_tree.branches]).numpy())
                    candidate.append(curr_tree.branches[first_choice].bottom.hit)
                    curr_layer = curr_tree.coord[0] + 1 #curr_tree.branches[first_choice].bottom.coord[0]
                    while True:
                        if candidate[-1] is None:
                            break
                        elif curr_layer >= event.shape[0] - 1:
                            candidate.append(None)
                            break
                        next_layer_hits = event[curr_layer+1].to_tensor()
                        curr_hit_expanded = tf.repeat(tf.expand_dims(candidate[-1], 0), next_layer_hits.shape[0], 0)
                        distr_input = tf.concat([curr_hit_expanded, next_layer_hits], 1)
                        distr = tf.reshape(Policy(distr_input), [-1])
                        distr_reduced = distr / tf.reduce_sum(distr)
                        mc_choice = np.random.choice(event[curr_layer+1].shape[0], p=reduce(distr).numpy())
                        if mc_choice == 0:
                            candidate.append(None)
                        else:
                            candidate.append(event[curr_layer+1,mc_choice])
                        curr_layer += 1
                
                # Backpropagation
                assert candidate[-1] is None and candidate[-2] is not None
                quality = Evaluate(tf.expand_dims(tf.stack(candidate[:-1]), 0))[0,0].numpy()
                if first_choice is not None:
                    curr_tree.branches[first_choice].update(quality)
                while curr_tree.parent is not None:
                    curr_tree.parent.update(quality)
                    curr_tree = curr_tree.parent.top

            # Execute best choice
            best_choice = np.argmax([branch.action_val() for branch in hit_tree.branches])
            if best_choice == 0:
                break
            else:
                track.append(hit_tree.branches[best_choice].bottom.hit)
                truth_track.append(hit_tree.branches[best_choice].bottom.truth)
            hit_tree = hit_tree.branches[best_choice].bottom
            hit_tree.parent = None
    
        tracks.append(track)
        truth_tracks.append(truth_track)

    return tracks if truth is None else (tracks, truth_tracks)

def track_metrics(truth_set, keys):
    eff_sum = 0
    purity_sum = 0
    count = 0

    for event_no, truth_tracks in enumerate(truth_set):
        total_frequency = {}
        for layer_no in range(keys[event_no].shape[0]):
            for hit_no in range(keys[event_no][layer_no].shape[0]):
                particle_id = keys[event_no][layer_no,hit_no].numpy()
                if particle_id in total_frequency:
                    total_frequency[particle_id] += 1
                else:
                    total_frequency[particle_id] = 1
        print(total_frequency)
        for track in truth_tracks:
            try:
                count += 1

                mode_id_result = stats.mode(track)
                mode_id = mode_id_result.mode[0]
                match_count = mode_id_result.count[0]
                purity_sum += match_count / len(track)

                total_count = total_frequency[mode_id]
                eff_sum += match_count / total_count
            except:
                print("Not finished", count)
                return eff_sum / count, purity_sum / count

    return eff_sum / count, purity_sum / count

def main(argv):
    hit_data_path = "smeared_hits.hdf5"
    policy_model_path = "saved_models/model_policy_nt_0.0365"
    eval_model_path = "saved_models/model_evaluate_nt_2epoch"
    nparticles = 8

    Policy = keras.models.load_model(policy_model_path)
    Evaluate = keras.models.load_model(eval_model_path)

    train_data, train_key, test_data, test_key, key_lst = get_telescope_layer_data(hit_data_path, 0.8, nparticles, hit_info=["tx", "ty", "tz"])
    print(key_lst)
    print(train_data.shape)
 
    naive_truth_sets = []
    truth_sets = []
    for target in range(test_data.shape[0]):
        if target % 10 == 0:
            print("Processed:", target)
        naive_tracks, naive_truth_tracks = naive_tracking(test_data[target], Policy, test_key[target])
        tracks, truth_tracks = tracking(test_data[target], Policy, Evaluate, test_key[target])
        naive_truth_sets.append(naive_truth_tracks)
        truth_sets.append(truth_tracks)

    naive_eff, naive_pur = track_metrics(naive_truth_sets, test_key)
    efficiency, purity = track_metrics(truth_sets, test_key)
    print()
    print("Test events:\t\t", len(truth_sets))
    print()
    print("Naive efficiency:\t\t", naive_eff)
    print("Naive purity:\t\t", naive_pur)
    print()
    print("MCTS efficiency:\t\t", efficiency)
    print("MCTS purity:\t\t", purity)
    
    print()
    print("Finished!")

if __name__ == "__main__":
    main(sys.argv)
