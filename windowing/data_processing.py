import numpy as np
import torch
from tqdm.notebook import tqdm

def split_sequences(data, sequence_length, fft=True):    
    # Split data into time, features, and labels
    time = data.values[:, 0]  # time column to partition data by
    X = data.values[:, 1:-1]  # EMG channels
    y = data.values[:, -1].astype(np.int64)  # class labels

    # Lists to store the sequences and corresponding labels
    X_sequences = []
    y_labels = []
    
    # Initialize start of the segment
    start_idx = 0
    
    # Determine target length after FFT and padding
    target_length = sequence_length // 2 + 1 if fft else sequence_length
    if target_length % 2 != 0:
        target_length += 1  # Ensure the length is even
    
    for i in tqdm(range(1, len(data))):
        # Check if the time has reset (indicating a new test segment)
        if time[i] < time[i - 1]:
            # Process the current segment up to this point
            for j in range(start_idx, i - sequence_length + 1):
                end_idx = j + sequence_length
                # Collect the sliding window and the label at the end of the window
                group = torch.tensor(X[j:end_idx], dtype=torch.float32)
                if fft:
                    group = torch.fft.rfft(group, dim=0)
                    magnitude = torch.abs(group)
                    phase = torch.angle(group)
                    group = torch.cat([magnitude, phase], dim=1)  # Concatenate magnitude and phase
                    # Pad to target length if necessary
                    if group.size(0) < target_length:
                        group = torch.cat([group, torch.zeros(target_length - group.size(0), group.size(1), dtype=group.dtype)], dim=0)
                X_sequences.append(group)
                y_labels.append(torch.tensor(y[end_idx - 1], dtype=torch.long))
            # Update start of the new segment
            start_idx = i
    
    # Process the final segment
    for j in range(start_idx, len(data) - sequence_length + 1):
        end_idx = j + sequence_length
        group = torch.tensor(X[j:end_idx], dtype=torch.float32)
        
        if fft:
            group = torch.fft.rfft(group, dim=0)
            magnitude = torch.abs(group)
            phase = torch.angle(group)
            group = torch.cat([magnitude, phase], dim=1)  # Concatenate magnitude and phase
            # Pad to target length if necessary
            if group.size(0) < target_length:
                group = torch.cat([group, torch.zeros(target_length - group.size(0), group.size(1), dtype=group.dtype)], dim=0)
                #group[torch.isnan(group)] = 0  # replace NaN's with zero
        
        X_sequences.append(group)
        y_labels.append(torch.tensor(y[end_idx - 1], dtype=torch.long))
    
    # Stack the lists into tensors
    X = torch.stack(X_sequences)
    y = torch.stack(y_labels)
    
    return X, y