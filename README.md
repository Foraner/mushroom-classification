
Mushroom Classification with Neural Networks & PCA
A binary classification model that predicts whether a mushroom is edible or poisonous based on physical characteristics. Built with TensorFlow/Keras. The main goal was to see how much PCA could reduce the feature space without hurting accuracy — turns out, quite a bit.
Tools

Python (Pandas, NumPy)
Scikit-learn (PCA, StandardScaler, preprocessing)
TensorFlow / Keras
Matplotlib

Data

Source: UCI Mushroom Dataset
8,124 samples, 22 categorical features (cap shape, odor, gill size, etc.)
Target: edible (e) or poisonous (p)

Approach

One-hot encoded 22 categorical features → 116 binary columns
Standardized features with StandardScaler (required before PCA)
Built a feed-forward neural network — 16-neuron hidden layer with ReLU, sigmoid output
Applied PCA to reduce 116 features down to 60 while retaining 95% of the variance
Re-trained the same architecture on the reduced data to compare performance

Results
ModelAccuracyPrecisionRecallF1Full features (116)99.94%100%99.87%99.94%PCA-reduced (60)99.88%100%99.74%99.87%
Cut the feature space nearly in half, lost less than 0.1% accuracy. In a safety-critical context — where a false negative means someone eats a poisonous mushroom — both models performed well, but the full-feature model is the safer choice.
Key Takeaways

PCA combined 116 features into 60 new components while preserving 95% of the variance
StandardScaler is essential before PCA — without it, results are misleading because PCA is sensitive to feature scale
The tradeoff between efficiency and accuracy was minimal here, but with larger datasets the computational savings would matter more

How to Run

Clone this repo
Install dependencies:

bash   pip install -r requirements.txt

Open the notebook and run cells top to bottom


Patrick Foran — LinkedIn · patrickmforan@gmail.com
