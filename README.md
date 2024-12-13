![banner](https://github.com/PJURC-data-science/red-wine-quality/blob/main/media/banner.png)

# Red Wine Quality Prediction: Property-Based Quality Analysis
[View Notebook](https://github.com/PJURC-data-science/red-wine-quality/blob/main/Red%20Wine%20Quality.ipynb)

A machine learning analysis of red wine properties to predict quality ratings and identify key quality indicators. This study employs multiple ML models to establish relationships between chemical properties and wine quality scores.

## Overview

### Business Question 
Which chemical properties most significantly influence red wine quality, and can these properties reliably predict quality ratings?

### Key Findings
- Most features show non-normal distribution
- Seven key predictive properties identified
- Random Forest outperforms Ordinal Logistic Regression
- Model predicts classes 5-6 most accurately
- Prediction errors tend toward higher ratings

### Impact/Results
- Identified key quality predictors
- Developed prediction model
- Established property benchmarks
- Quantified feature importance
- Created quality assessment framework

## Data

### Source Information
- Dataset: Red Wine Quality Dataset
- Source: UCI Machine Learning Repository
- Size: ~2000 wine samples
- Year: 2009

### Variables Analyzed
- Volatile acidity
- Citric acid
- Chlorides
- Total sulfur dioxide
- pH
- Sulphates
- Alcohol
- Quality ratings

## Methods

### Analysis Approach
1. Exploratory Analysis
   - Distribution testing
   - Correlation analysis
   - Outlier detection
2. Feature Selection
   - LASSO regression
   - Correlation significance
   - Variable importance
3. Model Development
   - Random Forest
   - Ordinal Logistic Regression
   - Grid Search optimization

### Tools Used
- Python (Data Science)
  - Numpy: Numerical operations
  - Pandas: Data manipulation
  - Scipy: Statistical testing
  - Seaborn/Matplotlib: Visualization
  - Scikit-learn:
    - Random Forest Classifier
    - Logistic Regression
    - LassoCV
    - GridSearchCV
    - Train-test splitting
    - Standard scaling
    - Performance metrics (R², MSE, MAE)
    - Confusion matrix
- Tableau (Interactive Data Visualization)

## Getting Started

### Prerequisites
```python
ipython==8.12.3
matplotlib==3.8.4
numpy==2.2.0
pandas==2.2.3
scikit_learn==1.6.0
scipy==1.14.1
seaborn==0.13.2
utils==1.0.2
```

### Installation & Usage
```bash
git clone git@github.com:PJURC-data-science/red-wine-quality.git
cd red-wine-quality
pip install -r requirements.txt
jupyter notebook "Red Wine Quality.ipynb"
```

## Project Structure
```
red-wine-quality/
│   README.md
│   requirements.txt
│   Red Wine Quality.ipynb
|   utils.py
└── data/
    └── winequality-red.csv
```

## Strategic Recommendations
1. **Quality Control**
   - Monitor key chemical properties
   - Focus on identified predictors
   - Implement testing protocols
   - Track quality consistency

2. **Production Guidelines**
   - Optimize alcohol content
   - Control acidity levels
   - Manage sulfur dioxide
   - Balance pH levels

3. **Quality Assessment**
   - Use prediction model
   - Monitor key indicators
   - Track quality distribution
   - Validate predictions

## Future Improvements
- Expand dataset size
- Add categorical modeling
- Enhance grid search
- Test GBM models
- Update data recency
- Validate quality calculations