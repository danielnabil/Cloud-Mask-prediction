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
- Set **Accelerator** to **TPU** under the "Notebook Settings"

### 2. Add Files

- Add the inference script file `evaluation_function.py` in the first cell
- In a second cell, add the `run_inference.py`  file

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

