# Data

This project uses the **UCI Bank Marketing dataset**, which contains records from a Portuguese bank's direct marketing campaigns (phone calls). The task is binary classification: predicting whether a client subscribed to a term deposit.

## Download Instructions

1. Go to https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Click "Download" to get the zip file
3. Extract `bank-full.csv` into this folder

After extraction, the file should be at `data/bank-full.csv`.

## Dataset Details

- **Records**: 45,211
- **Features**: 16 input features plus 1 target
- **Target**: `y` (binary: yes/no for term deposit subscription)
- **Baseline conversion rate**: about 11.7 percent
- **Separator**: semicolon (`;`)

## Feature Reference

**Client attributes**
- `age`: numeric
- `job`: type of job (categorical)
- `marital`: marital status (categorical)
- `education`: education level (categorical)
- `default`: has credit in default (yes/no)
- `balance`: average yearly balance in euros (numeric)
- `housing`: has housing loan (yes/no)
- `loan`: has personal loan (yes/no)

**Last contact attributes**
- `contact`: contact type (cellular/telephone/unknown)
- `day`: last contact day of month
- `month`: last contact month
- `duration`: last contact duration in seconds (numeric)

**Campaign attributes**
- `campaign`: number of contacts during this campaign
- `pdays`: days since last contact in previous campaign
- `previous`: number of contacts before this campaign
- `poutcome`: outcome of previous campaign

**Target**
- `y`: has the client subscribed to a term deposit (yes/no)
