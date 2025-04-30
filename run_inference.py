import os
import random
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
from evalution_function import rle_encode, rle_decode, dice_coefficient

# Initialize TPU if present
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(e)
    print("Running on CPU/GPU")
    
BATCH_SIZE = 8 * strategy.num_replicas_in_sync  # adjust batch size as needed
IMG_SIZE = 256  # image height and width
# Set seed for reproducibility
SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define paths (using your cloud masking dataset on Kaggle)
DATASET_DIR = Path("/kaggle/input/cloud-masking-test-set-satellite-cmp25-course")
IMGS_DIR = DATASET_DIR / "test/test/data"

image_test_paths = sorted(list(IMGS_DIR.glob("*.tif")))
print(f"Total images found: {len(image_test_paths)}")



def read_tif_image(file_path):
    def _read(path):
        im = tifffile.imread(path.decode('utf-8'))
        return im.astype('float32')
    img = tf.numpy_function(_read, [file_path], tf.float32)
    # Now img is [H, W, 4]
    img = tf.ensure_shape(img, [None, None, 4])
    return img / 255.0
    

def parse_image_with_id(img_path):
    image = read_tif_image(img_path)  # [H, W, 4]

    # Resize both to (IMG_SIZE, IMG_SIZE)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method='bilinear')

    img_path_str = tf.strings.as_string(img_path)
    image_id = tf.strings.split(img_path_str, os.sep)[-1]  # Get filename
    image_id = tf.strings.split(image_id, '.')[0]  # Remove extension

    return image,image_id


def get_dataset_inference(img_files, batch_size, augment=False, parse_fn=parse_image_with_id):
    ds = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in img_files])
    )
    ds = ds.map(lambda i: parse_fn(i), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000, seed=SEED)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


image_tests = image_test_paths
image_test = list(image_tests)

# Print sizes for test cell
print(f"Test set size: {len(image_test)}")

test_ds_id = get_dataset_inference(image_test, BATCH_SIZE, augment=False, parse_fn=parse_image_with_id)


def dice_coe(y_true, y_pred, smooth=1e-7):
    """
    Compute mean Dice coefficient per sample, matching dice_coefficient logic.
    Args:
        y_true: Ground truth masks, shape (batch_size, IMG_SIZE, IMG_SIZE, 1) or (batch_size, IMG_SIZE, IMG_SIZE).
        y_pred: Predicted masks, same shape as y_true.
        smooth: Smoothing factor (1e-7 to match dice_coefficient).
    Returns:
        Mean Dice coefficient across the batch.
    """
    # Ensure inputs are float32 and binarize predictions
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)  # Threshold at 0.5

    # Compute intersection and sums per sample
    # Sum over spatial dimensions (handles both 2D and 3D masks)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3] if y_true.shape.rank == 4 else [1, 2])
    sum_true = tf.reduce_sum(y_true, axis=[1, 2, 3] if y_true.shape.rank == 4 else [1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2, 3] if y_true.shape.rank == 4 else [1, 2])

    # Compute Dice coefficient per sample
    dice = (2.0 * intersection) / (sum_true + sum_pred + smooth)

    # Return mean Dice across the batch
    return tf.reduce_mean(dice)


# Initialize lists to store predictions, ground truth, and IDs
y_predictions = []
y_truth = []
ids = []
# Inference loop
with strategy.scope():
    # model = tf.keras.models.load_model("model_best_combined.keras", custom_objects={"dice_coe": dice_coe, "combined_loss": combined_loss})
    model = tf.keras.models.load_model("/kaggle/input/customunet/tensorflow2/default/1/CustomUnet.keras", custom_objects={"dice_coe": dice_coe})
for batch in test_ds_id:
    x, batch_ids = batch  # Unpack images, masks, and IDs
    preds = model.predict(x)  # Predict masks
    preds = tf.cast(tf.greater(preds, 0.5), tf.float32)  # Threshold predictions at 0.5
    y_predictions.append(preds.numpy())  # Convert to NumPy
    # print(batch_ids.numpy())
    ids.extend([f"{int(id.decode('utf-8')):06d}" for id in batch_ids.numpy()])


y_predictions = np.concatenate(y_predictions, axis=0)


y_submission = y_predictions

rle_submission = [rle_encode(mask) for mask in y_submission]

submission_df = pd.DataFrame({
    'id': ids,
    'segmentation': rle_submission
})

# Save to CSV
submission_df.to_csv('team_17.csv', index=False)
print("submission.csv created.")

# When reading back, preserve leading zeros
submission_df = pd.read_csv('team_17.csv', dtype={'id': str})
print(submission_df.head(5))

