# Mushroom Classification with Neural Networks

## Project Overview
This project explores whether mushrooms are edible or poisonous using a simple neural network built with **TensorFlow/Keras**.  
I trained two versions of the model — one using raw one-hot encoded data and another using **PCA** for dimensionality reduction — to see how much efficiency could be gained without sacrificing accuracy.

## Tools & Libraries
- Python (Pandas, NumPy)
- Scikit-learn (PCA, preprocessing)
- TensorFlow / Keras (neural network)
- Matplotlib (visualization)

## Data
- Source: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)  
- Features include cap shape, color, odor, gill size, and other visible traits.  
- The target variable is binary: **edible (e)** or **poisonous (p)**.  

## Approach
1. Cleaned and organized the categorical data, then label-encoded and one-hot encoded features.  
2. Built a basic feed-forward neural network on the one-hot encoded data.  
3. Applied **Principal Component Analysis (PCA)** to reduce the feature space while keeping 95% of the variance.  
4. Re-trained the same model architecture on the PCA data to compare accuracy, recall, precision, and training time.

## Results
**Baseline (One-Hot Encoded):** Accuracy **0.9994** | Precision **1.0000** | Recall **0.9987** | F1 **0.9994**  
**PCA (95% Variance):** Accuracy **0.9988** | Precision **1.0000** | Recall **0.9974** | F1 **0.9987**

Both models performed almost perfectly. The PCA version trained slightly faster, showing that dimensionality reduction can improve efficiency without losing performance.

### Confusion Tables
**Baseline (OHE)**
```
Predicted    0    1
Actual             
0          842    0
1            1  782
```

**PCA (95% var)**
```
Predicted    0    1
Actual             
0          842    0
1            2  781
```

##  Takeaways
- Solidified my understanding of end-to-end model building — from data cleaning to evaluation.  
- PCA effectively cut the features in half (≈116 → 60) while maintaining 95% of the dataset’s variance.  
- Reinforced practical experience with **TensorFlow/Keras**, **Scikit-learn**, and **Pandas** for applied machine learning.

##  What’s Next
- Experiment with deeper architectures or regularization for fine-tuning.  
- Package the notebook into a script or lightweight Flask app.  
- Try other dimensionality-reduction methods like t-SNE or UMAP for visualization and clustering.  

## How to Run
1. Clone this repo.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook and run the cells from top to bottom.

---

**Author:** Patrick Foran  
**Contact:** [LinkedIn](https://www.linkedin.com/in/patrickmforan) · patrickmforan@gmail.com
