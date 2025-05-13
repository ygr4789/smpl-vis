import numpy as np

def format_joint_sequences(*sequences):
    """
    Format multiple joint sequences into the expected npy format.
    
    Args:
        *sequences: Variable number of numpy arrays, each of shape (si, 24, 3)
                   where si is the sequence length for sequence i
    
    Returns:
        Dictionary in the format expected by the visualization code
    """
    num_seqs = len(sequences)
    seq_lengths = [seq.shape[0] for seq in sequences]
    
    max_len = max(seq_lengths)
    combined_seq = np.zeros((num_seqs, 24, 3, max_len))
    
    for i, seq in enumerate(sequences):
        combined_seq[i, :, :, :seq_lengths[i]] = seq.transpose(1, 2, 0)[:, :, :seq_lengths[i]]
    
    data_dict = {
        'motion': combined_seq,
        'num_samples': num_seqs,
    }
    
    return data_dict