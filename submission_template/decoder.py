"""Your decoder for the code check.

The top 10 teams hand in this folder: decoder.py, requirements.txt and any files decode() loads, together with
the code that created them. We call decode() once per basis on that basis's 100,000 shots in data/kaggle_private and time the two
calls (see "Handing in your code" in the README).

Everything that happens inside decode() counts towards your runtime, including building matrices from the circuit
or loading trained weights. Training doesn't count, but it must be reproducible from the code you hand in.
"""
import numpy as np
import stim

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
    base = "X"
        

    #convert the detector events into strings of lengths: 7,14,14,14,14,7
    detector_strings_raw = np.zeros((detector_events.shape[0], 10), dtype=np.uint8)
    for i in range(10):
        detector_strings_raw[:, i] = np.packbits(detector_events[:, i * 7:(i + 1) * 7], axis=1).flatten()

    #detectorstrings = np.zeros((detector_events.shape[0], 6), dtype=np.uint8)
    if base == "X":
        detector_strings = detector_strings_raw[:, [0, 2, 4, 6, 8, 9]]
    else:
        detector_strings = detector_strings_raw[:, [1, 3, 5, 7, 8, 9]]

    #print(detector_strings[0])    

    #1
    #priors, tabulated
    prior_spatialdistance = {0: 0.794472, 4: 0.014152, 1: 0.102598, 5: 0.005938, 3: 0.032914, 2: 0.04787, 6: 0.001948, 7: 0.000108}
    prior_velocity = {0: 0.47411, 2: 0.151165, 1: 0.2061275, 3: 0.117615, 4: 0.03862, 5: 0.010235, 6: 0.0019975, 7: 0.00013}
    #format these as arrays:
    prior_spatialdistance = np.array([0.794472, 0.014152, 0.102598, 0.005938, 0.032914, 0.04787, 0.001948, 0.000108])
    prior_velocity = np.array([0.47411, 0.151165, 0.2061275, 0.117615, 0.03862, 0.010235, 6: 0.0019975, 7: 0.00013}

    syndrome_history = detector_strings.copy() #initial syndrome history guess
    #print(syndrome_history[:4,:]) #print the first 4 syndrome histories
    proposed_syndrome_history = np.zeros_like(syndrome_history) #to store the updates to the syndrome history
    acceptance_probabilities = np.zeros(syndrome_history.shape[0],float) #to store the acceptance probabilities for each shot
    
    #now, propose a syndrome history. Use a metropolis-hastings algorithm to propose various different syndrome histories, and use the priors to make an accept function, taking into account detector events
    def acceptfunction(proposed, detector_strings, prior_spatialdistance, prior_velocity):
        #calculate the probability of the syndrome history given the priors and the detector events
        #first, compute the hamming distance (note that they are in integer form)
        hamming_distances_1 = np.zeros(proposed.shape[0], dtype=np.uint8)
        for i in range(proposed.shape[1]):
            var = proposed[:, i] ^ detector_strings[:, i]
            unpacked = np.unpackbits(var[:, np.newaxis], axis=1)
            summation = unpacked.sum(axis=1)
            hamming_distances_1 += summation
        #second, compute the hamming distances for the history updates
        hamming_distances_2 = np.zeros((proposed.shape[0],proposed.shape[1]-1), dtype=np.uint8)
        for i in range(proposed.shape[1]-1):
            var = proposed[:, i] ^ proposed[:, i+1]
            unpacked = np.unpackbits(var[:, np.newaxis], axis=1)
            summation = unpacked.sum(axis=1)
            hamming_distances_2[:, i] = summation
        #now, compute the probabilities given the priors
        probs1 = np.array([prior_spatialdistance[hd] for hd in hamming_distances_1])#!make this smarter later
        probs2 = np.array([prior_velocity[hd] for hd in hamming_distances_2])#!make this smarter later
        return np.prod(probs1,axis=1) * np.prod(probs2,axis=1) #return the product of the probabilities for each shot
    number_of_MCMC_steps = 100
    number_of_historypoints = 6
    MC_outcomes = np.zeros((number_of_MCMC_steps, syndrome_history.shape[0], number_of_historypoints), dtype=np.uint8) #to store the outcomes of the MCMC steps
    number_of_bits_per_historypoint = 7
    previous_probabilities = np.zeros(syndrome_history.shape[0], float)
    for n_MCMC in range(number_of_MCMC_steps):
        for t in range(number_of_historypoints):
            for b in range(number_of_bits_per_historypoint):
                #propose a new syndrome history by flipping a bit
                proposed_syndrome_history = syndrome_history.copy()
                proposed_syndrome_history[:, t] ^= (1 << b)
                print(proposed_syndrome_history[:3,:]) #print the first 3 proposed syndrome histories
                #calculate the acceptance probability
                probability = acceptfunction(proposed_syndrome_history, detector_strings, prior_spatialdistance, prior_velocity)
                acceptance_probabilities = np.minimum(1, probability / previous_probabilities)
                #accept or reject the proposed syndrome history
                random_numbers = np.random.rand(syndrome_history.shape[0])
                mask = random_numbers < acceptance_probabilities
                syndrome_history[mask] = proposed_syndrome_history[mask]
                MC_outcomes[n_MCMC] = syndrome_history
                previous_probabilities = acceptance_probabilities
    
    return np.zeros((len(detector_events), vectorsize), dtype=np.uint8)


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """
    circuit: the X or Z circuit, loaded with stim.Circuit.from_file.
    detector_events: bool array of shape (shots, 70), columns D0 to D69.
    Returns a 0/1 array of shape (shots, 4), columns L0 to L3.
    """
    #the decoding will be done in 2 stages. First, a syndrome string is predicted. After that, a probable logical error is inferred
    
    syndrome_estimates = data2syndrome(detector_events)

    # Replace this with your decoder. Predicting no flips gives an LER of about 0.29.
    return np.zeros((len(detector_events), 4), dtype=np.uint8)
