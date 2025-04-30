import numpy as np
import pandas as pd
import os

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    
    Returns:
        str: RLE-encoded string.
    """
    if np.sum(mask) == 0: return " "
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format

class ParticipantVisibleError(Exception):
    """ Custom exception shown to participants. """
    pass

def rle_decode(mask_rle: str, shape=(256, 256)) -> np.ndarray:
    """Decodes an RLE-encoded string into a binary mask with validation checks."""
    
    if not isinstance(mask_rle, str) or not mask_rle.strip() or mask_rle.lower() == 'nan':
        # Return all-zero mask if RLE is empty, invalid, or NaN
        return np.zeros(shape, dtype=np.uint8)
    
    try:
        s = list(map(int, mask_rle.split()))
    except:
        raise ParticipantVisibleError("RLE segmentation must contain only integers")
    
    if len(s) % 2 != 0:
        raise ParticipantVisibleError("RLE segmentation must have even-length (start, length) pairs")
    
    if any(x < 0 for x in s):
        raise ParticipantVisibleError("RLE segmentation must not contain negative values")
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    starts, lengths = s[0::2], s[1::2]
    
    for start, length in zip(starts, lengths):
        if start >= mask.size or start + length > mask.size:
            raise ParticipantVisibleError("RLE indices exceed image size")
        mask[start:start + length] = 1
    
    return mask.reshape(shape, order='F')  # Convert to column-major order

def dice_coefficient(mask1, mask2):
    """ Computes Dice coefficient between two binary masks. """
    intersection = np.sum(mask1 * mask2)
    return (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-7)  # Add epsilon to avoid zero division

def evaluate_submission(solution_df, submission_df):
    """ Scores the submission against the solution using Dice coefficient. """
    
    # Validate columns
    required_cols = {"id", "segmentation"}
    if not required_cols.issubset(solution_df.columns) or not required_cols.issubset(submission_df.columns):
        raise ParticipantVisibleError("Both solution and submission must contain 'id' and 'segmentation' columns.")
    
    # Validate ID order
    if not solution_df['id'].equals(submission_df['id']):
        raise ParticipantVisibleError("Submission IDs do not match solution IDs.")
    
    # Drop id columns
    solution_masks = solution_df['segmentation']
    submission_masks = submission_df['segmentation']
    
    dice_scores = []
    for sol_mask_rle, sub_mask_rle in zip(solution_masks, submission_masks):
        sol_mask = rle_decode(sol_mask_rle)
        sub_mask = rle_decode(sub_mask_rle)
        dice_scores.append(dice_coefficient(sol_mask, sub_mask))
    
    return np.mean(dice_scores)