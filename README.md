# Cloud Masking Inference Guide (Kaggle TPU)

This repository provides instructions to run inference on satellite images using our pre-trained CustomUNet model. We utilize **TPU acceleration on Kaggle** to ensure fast and efficient processing.

---

## 📦 Requirements

### ✅ On Kaggle (TPU)

All required packages and libraries are pre-installed in the Kaggle TPU environment. You **only** need to install the `tifffile` library:

```python
!pip install tifffile
```
### 💻 Running Locally

If you prefer to run the inference code on your own machine, please make sure you have the following packages installed:
```bash
pip install tensorflow keras tifffile
```

---

## ⚡ Quick Start on Kaggle (TPU Inference)

Follow these steps to run `run_inference.py` using Kaggle with TPU enabled:

### 1. Create a New Notebook

- Go to [Kaggle Notebooks](https://www.kaggle.com/code)
- Click **"New Notebook"**
- Set **Accelerator** to **TPU OR GPU P100** if TPU is overloaded under the "Notebook Settings"

### 2. Add Files

- Add the inference script file `evaluation_function.py` in the first cell
- In a second cell, add the `run_inference.py`  file don't forget to remove ```from evalution_function import rle_encode, rle_decode, dice_coefficient``` line
- You could add this code in a seperate cell for more stable kaggle enviornment
```python
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

```
# Initialize TPU if present You could also use GPU 100 it will be good for inference 
### 3. Attach Required Datasets and Model

- Attach the test dataset (TA Dataset):  
  🔗 [Cloud Masking Test Set - Satellite CMP25 Course](https://www.kaggle.com/datasets/nouranhany10/cloud-masking-test-set-satellite-cmp25-course)

- Attach the pre-trained model (CustomUNet):  
  🔗 [CustomUNet - Gousha Model](https://www.kaggle.com/models/goushaa/customunet)

> 📌 **Transparency Notice**:  
> The model was created before the submission deadline for full reproducibility and fairness.

### 4. Run the Notebook

- Run the **first cell** containing `evaluation_function.py`
- Run the **second cell** containing `run_inference.py` 

### 5. Output

- The script will export the results as a CSV file named:  
  📄 `team_17.csv`

---

