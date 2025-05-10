# Subjective Poverty Level Prediction

## Project Overview
This project focuses on predicting subjective poverty levels using machine learning techniques. By analyzing various socioeconomic indicators, we aim to model and predict how individuals perceive their own economic status.

## Dataset Description
The dataset includes various features related to household characteristics and economic indicators. The target variable is the subjective poverty score, which ranges from 1 to 10, with 1 being the poorest and 10 being the richest. 

### Data Distribution
The training set shows a skewed normal distribution of subjective poverty scores centered around 4-5, with very few observations in the range of 9 to 10. The skewed distribution of
the data was an important consideration that had to be accounted for in the modelling stage.

![Rplot](https://github.com/user-attachments/assets/0ea6f3d2-b36b-4c04-8e99-51393f29cbe1)


## Feature Importance
Our analysis revealed that several key features have significant predictive power in determining subjective poverty levels. The "psu" (Primary Sampling Unit) feature showed the highest importance.

![output](https://github.com/user-attachments/assets/bab2a639-edeb-4030-ac84-877f7ae17569)

## Model Development
We experimented with multiple machine learning algorithms, with a focus on ensemble methods that handle complex relationships in socioeconomic data effectively.

### Model Performance
Two main models were evaluated:

1. **XGBoost**
   - Initial error rate: ~2.0
   - Final training error: ~0.6
   - Final test error: ~1.6
   - The model showed significant improvement during training but revealed some overfitting as iterations increased.

![Rplot01](https://github.com/user-attachments/assets/8bc5e60d-d88b-41d8-91fd-73af8d5f8c8f)

2. **Random Forest**
   - Initial error rate: ~2.1
   - Final training error: ~0.8
   - Final test error: ~1.9
   - While slightly less prone to overfitting than XGBoost, the Random Forest model still showed a gap between training and test performance.

![Rplot02](https://github.com/user-attachments/assets/bae7a04e-3ca7-461f-8b6a-cbdb99c48ab4)


## Feature Engineering
After finding limitations in our predictive models, we decided to engineer features based on most significant variables to leverage their predictive power in improving our accuracy. We began experimenting with the Household ID (HHID), an ID related to Personal Sampling Unit (PSU) which was shown to be highly significant. To create a new variable we sorted our training dataset by HHID and added a rolling average of subjective poverty to each row. The rolling average was created by taking the 5 rows before and after each row in the sorted training dataset and averaging the subjective poverty from each of these rows. We thought the rolling average variable would aid our predictive accuracy reasoning that people in the same household or adjacent households would be likely to have a similar perspective on their subjective poverty. We also added upper and lower bound variables that showed the highest and lowest subjective poverty level within 5 adjacent rows of each row. 


<img width="477" alt="Picture1" src="https://github.com/user-attachments/assets/d1c9f8ad-def4-4f0f-a68c-e08d1ad7e56c" />


This graph shows:
- Blue dots: True subjective poverty levels
- Red line: Rolling average of subjective poverty level
- Green dashed line: Upper bound of subjective poverty in the 10 rows used to create the rolling average
- Orange dotted line: Lower bound of subjective poverty in the 10 rows used to create the rolling average

## Key Findings
1. Household identifiers and location-based features (psu, hhid) carry significant predictive power, suggesting strong regional or community-level factors in perceived poverty.
3. Both XGBoost and Random Forest models demonstrated similar learning patterns, with XGBoost performing slightly better on test data.
4. The models maintained a test error of approximately 1.6-1.9, suggesting that subjective poverty assessments have inherent variability that is challenging to predict with complete accuracy.

## Limitations and Future Work
1. **Overfitting**: Both models showed signs of overfitting, suggesting the need for stronger regularization or feature selection.
2. **Feature Engineering**: Further feature engineering based on domain knowledge could improve model performance.
3. **Alternative Models**: Exploring deep learning approaches or hybrid models might capture more complex patterns in the data.

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- Pandas
- Matplotlib
- NumPy
- R

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



## License
[Insert your chosen license here]

## Acknowledgements
[Include any acknowledgements, data sources, or inspirations]
