# Lead Scoring on UCI Bank Marketing Data

A 0-100 lead scoring model that predicts conversion likelihood for direct marketing campaigns. Built with logistic regression on the UCI Bank Marketing dataset.

This project demonstrates the methodology I used to build a production lead scoring system at F Street, where the model achieved AUC 0.84 on 3,000+ loan applications and was deployed in HubSpot as a 0-100 scoring framework. The work here uses public data so the code and weights can be shared openly.

## Why lead scoring matters

Most sales and marketing teams operate with too many leads and too little time. A lead scoring model ranks leads so the team can prioritize the leads most likely to convert. The lift comes from two things: identifying high-intent prospects early and deprioritizing leads unlikely to convert.

## Results

On the UCI Bank Marketing dataset (45K records, 11.7 percent baseline conversion):

| Metric | Value |
|---|---|
| Test AUC | 0.91 |
| 5-fold CV AUC | 0.89 +/- 0.01 |
| Top tier (80-100) conversion rate | ~55 percent |
| Bottom tier (0-19) conversion rate | ~2 percent |

The top decile converts at roughly 5x the baseline rate, meaning a sales team could capture the majority of conversions by focusing on the top 20-30 percent of leads.

## How it works

1. **Feature engineering** turns raw inputs into business-meaningful buckets (age brackets, balance tiers, contact intensity, prior outcome categories)
2. **Logistic regression** is trained on the engineered features with class balancing
3. **Score conversion** maps the predicted probability to a 0-100 score
4. **Tier assignment** groups scores into Hot, Warm, Lukewarm, Cold, and Frozen tiers
5. **Validation** uses AUC, decile lift, and per-tier conversion rates

Logistic regression was chosen over more complex models because the coefficients are interpretable, which matters for explaining the system to non-technical stakeholders and shipping it to a CRM.

## Limitations

Call duration is one of the strongest features, but it is only known after a call ends. The model therefore scores leads after first contact, not before it. A pre-contact version would need to drop duration, and would have a lower AUC.

## Project structure

```
Lead-Scoring-Bank-Marketing/
├── README.md
├── requirements.txt
├── data/
│   └── README.md              # Dataset download instructions
├── notebooks/
│   └── lead_scoring_analysis.ipynb
├── src/
│   ├── feature_engineering.py # Feature creation and bucketing
│   ├── model.py               # Model training pipeline
│   ├── scoring.py             # Probability to score conversion
│   ├── visualizations.py      # Chart generation
│   └── train.py               # End-to-end training script
├── app/
│   └── streamlit_app.py       # Interactive demo
└── outputs/
    ├── score_distribution.png
    ├── roc_curve.png
    ├── success_by_tier.png
    └── feature_importance.png
```

## Setup

```bash
git clone https://github.com/HarshetAnand/Lead-Scoring-Bank-Marketing.git
cd Lead-Scoring-Bank-Marketing
pip install -r requirements.txt
```

Download the UCI Bank Marketing dataset (see `data/README.md`) and place `bank-full.csv` in the `data/` folder.

## Usage

**Train the model and generate charts**

```bash
python src/train.py
```

This loads the data, trains the model, prints metrics, generates charts in `outputs/`, and saves the trained model as `outputs/model.pkl`.

**Explore the analysis**

```bash
jupyter notebook notebooks/lead_scoring_analysis.ipynb
```

**Run the interactive demo**

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app lets you input lead attributes and get a real-time score with tier classification.

## Sample output

After running the training script, you'll see output like:

```
Baseline conversion rate: 11.7%
Test set AUC: 0.907
5-fold CV AUC: 0.894 +/- 0.008

Conversion rate by tier:
            tier  leads  success_rate  avg_score
   Frozen (0-19)   5421           2.1       11.4
    Cold (20-39)   1894           7.8       28.7
Lukewarm (40-59)    867          18.4       48.2
    Warm (60-79)    524          37.1       68.5
    Hot (80-100)    337          54.6       88.1
```

## Key features

**Engineered features that drive lift**

- Prior campaign outcome (whether a previous campaign was successful)
- Contact intensity (combined campaign and previous contact counts)
- Call duration tiers (short calls indicate disengagement)
- Balance and age brackets aligned with customer segments

**Production-ready scoring layer**

- Score bucketing logic separated from the model for easy iteration
- Tier definitions that map directly to sales workflows
- Decile analysis for lift validation

## Relationship to my F Street work

At F Street, I built a lead scoring system from scratch for hard money loan applications. That system used the same approach: logistic regression on engineered features, AUC validation, score bucketing into a 0-100 framework, and HubSpot deployment for sales prioritization. It achieved AUC 0.84 on 3,000+ loan applications. A separate segmentation analysis identified the top 20 percent of borrowers driving roughly 30 percent of total loan volume.

This project rebuilds the methodology on public data so the code and weights can be shared. The features differ (loan applications vs term deposit campaigns) but the modeling framework is the same.

## Tech stack

- Python 3.10+
- scikit-learn for modeling
- pandas and numpy for data handling
- matplotlib for visualization
- Streamlit for the interactive demo

## License

MIT

## Author

**Harshet Anand**
UW-Madison '25, BS Computer Science and Data Science
[LinkedIn](https://linkedin.com/in/harshet-anand) | [GitHub](https://github.com/HarshetAnand)
