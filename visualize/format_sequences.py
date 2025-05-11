import numpy as np

def format_joint_sequences(seq1, seq2):
    """
    Format two joint sequences into the expected npy format.
    
    Args:
        seq1: numpy array of shape (s1, 24, 3)
        seq2: numpy array of shape (s2, 24, 3)
    
    Returns:
        Dictionary in the format expected by the visualization code
    """
    # Get sequence lengths
    s1 = seq1.shape[0]
    s2 = seq2.shape[0]
    
    # Combine sequences into a single array with shape (2, 24, 3, max(s1,s2))
    min_len = max(s1, s2)
    combined_seq = np.zeros((2, 24, 3, min_len))
    
    # Fill in the sequences
    combined_seq[0, :, :, :min_len] = seq1.transpose(1, 2, 0)[:, :, :min_len]  # (24, 3, s1)
    combined_seq[1, :, :, :min_len] = seq2.transpose(1, 2, 0)[:, :, :min_len]  # (24, 3, s2)
    
    # Create the data dictionary
    data_dict = {
        'motion': combined_seq,
        'num_samples': 2,
        'lengths': np.array([s1, s2]),
        'text': ['sequence_1', 'sequence_2']  # You can modify these descriptions
    }
    
    return data_dict

def save_sequences(seq1, seq2, output_path):
    """
    Save two joint sequences in the format expected by the visualization code.
    
    Args:
        seq1: numpy array of shape (s1, 24, 3)
        seq2: numpy array of shape (s2, 24, 3)
        output_path: path to save the formatted npy file
    """
    # Format the sequences
    data_dict = format_joint_sequences(seq1, seq2)
    
    # Save to npy file
    np.save(output_path, data_dict)
    print(f"Saved formatted sequences to {output_path}")

if __name__ == "__main__":
    import pickle
    
    # Load the pickle file
    with open('sample_41_seq_0_test.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    p1_jnts = data['full_refine_pred_p1_20fps_jnts_list']
    p2_jnts = data['full_refine_pred_p2_20fps_jnts_list']

    print("Original shapes:")
    print(f"p1_jnts: {p1_jnts.shape}")
    print(f"p2_jnts: {p2_jnts.shape}")

    # Save the sequences in the correct format
    output_path = 'formatted_sequences.npy'
    save_sequences(p1_jnts, p2_jnts, output_path) 