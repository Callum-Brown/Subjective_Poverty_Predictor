# Subjective Poverty Level Prediction

## Project Overview
This project focuses on predicting subjective poverty levels using machine learning techniques. By analyzing various socioeconomic indicators, we aim to model and predict how individuals perceive their own economic status.

## Dataset Description
The dataset includes various features related to household characteristics and economic indicators. The target variable is the subjective poverty score, which ranges from 1 to 10, with higher values indicating higher perceived poverty levels.

### Data Distribution
The training set shows a somewhat normal distribution of subjective poverty scores centered around 4-5, with fewer observations at the extreme ends of the scale.

![Distribution of labels in training set](https://github.com/yourusername/project-name/blob/main/images/label_distribution.png)

## Feature Importance
Our analysis revealed that several key features have significant predictive power in determining subjective poverty levels. The "psu" (Primary Sampling Unit) feature showed the highest importance, followed by various household-related indicators such as "q06_y" and "q04_x".

![Feature importance](https://github.com/yourusername/project-name/blob/main/images/feature_importance.png)

## Model Development
We experimented with multiple machine learning algorithms, with a focus on ensemble methods that handle complex relationships in socioeconomic data effectively.

### Model Performance
Two main models were evaluated:

1. **XGBoost**
   - Initial error rate: ~2.0
   - Final training error: ~0.6
   - Final test error: ~1.6
   - The model showed significant improvement during training but revealed some overfitting as iterations increased.

![XGBoost Learning Curve](https://github.com/yourusername/project-name/blob/main/images/xgboost_learning_curve.png)

2. **Random Forest**
   - Initial error rate: ~2.1
   - Final training error: ~0.8
   - Final test error: ~1.9
   - While slightly less prone to overfitting than XGBoost, the Random Forest model still showed a gap between training and test performance.

![Random Forest Learning Curve](https://github.com/yourusername/project-name/blob/main/images/random_forest_learning_curve.png)

## Validation Against Ground Truth
We validated our model predictions against the actual subjective poverty levels. The rolling average approach helped smooth out predictions and provide confidence intervals around our estimates.

![Rolling Average vs True Poverty Levels](https://github.com/yourusername/project-name/blob/main/images/rolling_average.png)

This graph shows:
- Blue dots: True subjective poverty levels
- Red line: Rolling average of predictions
- Green dashed line: Upper confidence bound
- Orange dotted line: Lower confidence bound

## Key Findings
1. Household identifiers and location-based features (psu) carry significant predictive power, suggesting strong regional or community-level factors in perceived poverty.
2. Questions related to household resources (q06_y, q04_x, q04_y) are strong predictors of subjective poverty assessment.
3. Both XGBoost and Random Forest models demonstrated similar learning patterns, with XGBoost performing slightly better on test data.
4. The models maintained a test error of approximately 1.6-1.9, suggesting that subjective poverty assessments have inherent variability that is challenging to predict with complete accuracy.

## Limitations and Future Work
1. **Overfitting**: Both models showed signs of overfitting, suggesting the need for stronger regularization or feature selection.
2. **Feature Engineering**: Further feature engineering based on domain knowledge could improve model performance.
3. **Alternative Models**: Exploring deep learning approaches or hybrid models might capture more complex patterns in the data.
4. **Temporal Analysis**: Incorporating time-series analysis could reveal how subjective poverty perceptions change over time.

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- Pandas
- Matplotlib
- NumPy

## How to Use This Repository
1. Clone the repository
   ```
   git clone https://github.com/yourusername/project-name.git
   ```
2. Install required dependencies
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks in the `notebooks/` directory to reproduce the analysis

## Project Structure
```
project-name/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── images/
│   ├── feature_importance.png
│   ├── label_distribution.png
│   ├── rolling_average.png
│   ├── xgboost_learning_curve.png
│   └── random_forest_learning_curve.png
├── requirements.txt
└── README.md
```

## License
[Insert your chosen license here]

## Acknowledgements
[Include any acknowledgements, data sources, or inspirations]