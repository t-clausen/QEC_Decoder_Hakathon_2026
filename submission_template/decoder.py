"""Your decoder for the code check.

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README).

Everything that happens inside decode() counts towards your runtime, including building matrices from the circuit
or loading trained weights. Training doesn't count, but it must be reproducible from the code you hand in.
"""
import numpy as np
import stim
import cupy as cp
from time import sleep  
from tqdm import tqdm

def syndrome2logical(last_syndrome:np.ndarray, basis:bool) -> np.ndarray:
    """
    last_syndrome: (shots, vectorsize) array of 7-bit syndrome history integers.
    Returns: (shots, 4) array of 0/1 logical error prediction probabilities.
    """
    # Replace this with your decoder. Predicting no flips gives an LER of about 0.29.
    return np.zeros((len(last_syndrome), vectorsize), dtype=np.uint8)

def data2syndrome(detector_events:np.ndarray, basis:bool) -> np.ndarray:
    """
    detector_events: (shots, 70) array of 0/1 detector events.
    Returns: (shots, 7) array of 7-bit syndrome history integers.
    """
        

    #convert the detector events into strings of lengths: 7,14,14,14,14,7
    detector_strings_raw = np.zeros((detector_events.shape[0], 10), dtype=np.uint8)
    for i in range(10):
        detector_strings_raw[:, i] = np.packbits(detector_events[:, i * 7:(i + 1) * 7], axis=1).flatten()

    #detectorstrings = np.zeros((detector_events.shape[0], 6), dtype=np.uint8)
    if not basis:
        detector_strings = detector_strings_raw[:, [0, 2, 4, 6, 8, 9]]
    else:
        detector_strings = detector_strings_raw[:, [1, 3, 5, 7, 8, 9]]

    #print(detector_strings[0])    

    #1
    #priors, tabulated
    prior_spatialdistance_X = np.array([0.794472, 0.014152, 0.102598, 0.005938, 0.032914, 0.04787, 0.001948, 0.000108,0])
    prior_spatialdistance_Z = np.array([0.90566, 0.072126, 0.013108, 0.00816, 0.000776, 0.000146, 2.4e-05,0])
    prior_velocity_X = np.array([0.47411, 0.151165, 0.2061275, 0.117615, 0.03862, 0.010235, 0.0019975, 0.00013,0])
    prior_velocity_Z = np.array([0.1958, 0.122265, 0.4838475, 0.146135, 0.000115, 0.0106075, 0.0390625, 0.0021675])
    if not basis:
        prior_spatialdistance = prior_spatialdistance_X
        prior_velocity = prior_velocity_X
    else:
        prior_spatialdistance = prior_spatialdistance_Z
        prior_velocity = prior_velocity_Z

    syndrome_history = detector_strings.copy() #initial syndrome history guess
    #print(syndrome_history[:4,:]) #print the first 4 syndrome histories
    proposed_syndrome_history = np.zeros_like(syndrome_history) #to store the updates to the syndrome history
    acceptance_probabilities = np.zeros(syndrome_history.shape[0],float) #to store the acceptance probabilities for each shot
    
    #now, propose a syndrome history. Use a metropolis-hastings algorithm to propose various different syndrome histories, and use the priors to make an accept function, taking into account detector events
    def acceptfunction(proposed, detector_strings, prior_spatialdistance_local, prior_velocity_X):
        #calculate the probability of the syndrome history given the priors and the detector events
        #first, compute the hamming distance (note that they are in integer form)
        hamming_distances_1 = np.zeros(proposed.shape, dtype=np.uint8)
        for i in range(proposed.shape[1]):
            var = proposed[:, i] ^ detector_strings[:, i]
            unpacked = np.unpackbits(var[:, np.newaxis], axis=1)
            summation = unpacked.sum(axis=1)
            hamming_distances_1[:, i] = summation
        #second, compute the hamming distances for the history updates
        hamming_distances_2 = np.zeros((proposed.shape[0],proposed.shape[1]-1), dtype=np.uint8)
        for i in range(proposed.shape[1]-1):
            var = proposed[:, i] ^ proposed[:, i+1]
            unpacked = np.unpackbits(var[:, np.newaxis], axis=1)
            summation = unpacked.sum(axis=1)
            hamming_distances_2[:, i] = summation
        #now, compute the probabilities given the priors
        #print(hamming_distances_1[:10]) #print the first 10 hamming distances for the first prior
        probs1 = prior_spatialdistance_local[hamming_distances_1]
        #compute the probabilities for the second prior (despite the funny shape array)
        probs2 = prior_velocity_X[hamming_distances_2]
        return np.prod(probs1,axis=1) * np.prod(probs2,axis=1) #return the product of the probabilities for each shot
    number_of_MCMC_steps = 10
    number_of_historypoints = 6
    MC_outcomes = np.zeros((number_of_MCMC_steps, syndrome_history.shape[0], number_of_historypoints), dtype=np.uint8) #to store the outcomes of the MCMC steps
    probabilities = np.zeros((number_of_MCMC_steps, syndrome_history.shape[0]), dtype=float) #to store the probabilities of the outcomes of the MCMC steps
    number_of_bits_per_historypoint = 7
    previous_probabilities = np.zeros(syndrome_history.shape[0], float)
    syndrome_probability_ditributions = np.zeros((syndrome_history.shape[0], 2**7), float) #to store the probability distributions of the final syndrome in the syndrome histories
    for n_MCMC in tqdm(range(number_of_MCMC_steps)):
        probs = np.zeros((number_of_historypoints, number_of_bits_per_historypoint, syndrome_history.shape[0]), float) #to store the probabilities of the proposed syndrome histories
        for t in range(number_of_historypoints):
            for b in range(number_of_bits_per_historypoint):
                #propose a new syndrome history by flipping a bit
                proposed_syndrome_history = syndrome_history.copy()
                proposed_syndrome_history[:, t] ^= (1 << b)
                #print(proposed_syndrome_history[:3,:]) #print the first 3 proposed syndrome histories
                #calculate the acceptance probability
                probability = acceptfunction(proposed_syndrome_history, detector_strings, prior_spatialdistance, prior_velocity)
                #print(probability[:3]) #print the first 3 probabilities
                acceptance_probabilities = np.minimum(1, probability / previous_probabilities)
                #accept or reject the proposed syndrome history
                random_numbers = np.random.rand(syndrome_history.shape[0])
                mask = random_numbers < acceptance_probabilities
                syndrome_history[mask] = proposed_syndrome_history[mask]
                previous_probabilities = acceptance_probabilities
                probs[t, b, :] = probability
                #print(syndrome_history[:10,:]) #print the first 10 syndrome histories
            temp = syndrome_history.flatten()
            temp[temp > 127] -= 127 #wrap around the syndrome history if it is greater than 127
            syndrome_history = temp.reshape(syndrome_history.shape)
        #MC_outcomes[n_MCMC] = syndrome_history
        #probabilities[n_MCMC] = acceptance_probabilities
        #print(f"MCMC step {n_MCMC+1}/{number_of_MCMC_steps} completed")
        #note the last syndrome, and the probability of this history to be corrrect
        prob = np.sum(probs, axis=(0, 1)) #sum over the history points and bits
        #print(syndrome_history[:10,:]) #print the first 10 syndrome histories
        last_syndrome = syndrome_history[:, -1]
        #last_syndrome[last_syndrome > 127] -= 127 #wrap around the last syndrome if it is greater than 127
        #print("last syndrome:", last_syndrome[:10])
        #print(prob[:10])
        #print(last_syndrome[:10])
        #use last_syndrome as indecies to add to syndrome_probability_ditributions array
        #print(syndrome_probability_ditributions.shape, last_syndrome.shape, prob.shape)
        syndrome_probability_ditributions[np.arange(syndrome_history.shape[0]), last_syndrome] += prob
    #do an average over these bit strings, weighted by their probabilities
    #print("syndrome probability distributions shape:", syndrome_probability_ditributions.shape)
    #print("syndrome probability distributions:", syndrome_probability_ditributions[:10,:])
    

    
    return syndrome_probability_ditributions

def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    if "31" in circuit[32]:
        basis = True    
    else:
        basis = False
    #the decoding will be done in 2 stages. First, a syndrome string is predicted. After that, a probable logical error is inferred
    
    syndrome_estimates = data2syndrome(detector_events, basis)

    # Replace this with your decoder. Predicting no flips gives an LER of about 0.29.
    return np.zeros((len(detector_events), 4), dtype=np.uint8)
