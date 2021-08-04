import numpy as np
from pathlib import Path
import pickle

num_stacked = 100 # the number of total images stacked on one another

# method to generate a proportion vector
def gen_prop_vec_unif():
  rand_vec = [np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked),
              np.random.random_integers(0, num_stacked)]
  rand_vec = np.round((rand_vec/np.sum(rand_vec))*num_stacked)
  if(np.sum(rand_vec) != num_stacked):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_stacked - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)

  return rand_vec

# method to generate a proportion vector
def gen_prop_vec_lognormal():
  rand_vec = np.random.lognormal(5, 3, 10)
  rand_vec = np.round((rand_vec/np.sum(rand_vec))*num_stacked)
  if(np.sum(rand_vec) != num_stacked):
    idx_change = np.argmax(rand_vec)
    rand_vec[idx_change] = rand_vec[idx_change] + (num_stacked - np.sum(rand_vec))

  rand_vec = rand_vec.astype(int)

  return rand_vec


# method to sample a single number many times
def gen_single_num_sum(num_samp, num_interest, X_in, Y_in):
  # first get the index for the number of interest
  interest_idx = np.where(Y_in == num_interest)

  # sample them num-samp times
  interest_idx = np.random.choice(interest_idx[0], num_samp, replace=False)
  X_interest = X_in[interest_idx,]
  X_sum = np.sum(X_interest, axis=0)
  return X_sum

# method to sum all the numbers in given proportions
def gen_prop_num_sum(prop_vec, X_in, Y_in):
  running_sum = None
  for curr_num in range(0, len(prop_vec)):
    num_samp = prop_vec[curr_num]
    curr_sum = gen_single_num_sum(num_samp, curr_num, X_in, Y_in)

    if running_sum is None:
        running_sum = curr_sum
    else:
        running_sum = curr_sum + running_sum

    # alternatively, the ternary operator, aka X if cond else Y
    # running_sum = cur_sum if running_sum is None else running_sum + cur_sum

  # alternatively, a list compehension
  # running_sum = np.sum([gen_single_num_sum(x, idx, X_in, Y_in) for x, idx in enumerate(prop_vec)])


  return running_sum / 100

def make_stacked_sample(X_in, Y_in):
  rand_vec = gen_prop_vec_lognormal()
  stacked_vec = gen_prop_num_sum(rand_vec, X_in, Y_in)

  return (rand_vec, stacked_vec)

def make_all_stacked_samples(X_in, Y_in, out_file, num_samples):
  
  out_path = Path(out_file)

  if out_path.is_file(): # load the data if we already generated it
    stacked = pickle.load( open( out_path, "rb" ) )
    Y_stack = np.stack(stacked[:,0])
    X_stack = np.stack(stacked[:,1])
    Y_stack = Y_stack/100
  else: # otherwise generate data
    stacked = np.stack([make_stacked_sample(X_in, Y_in) for x in range(0, num_samples)])
    Y_stack = np.stack(stacked[:,0])
    X_stack = np.stack(stacked[:,1])
    Y_stack = Y_stack/100

  return (X_stack, Y_stack)

