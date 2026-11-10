# Loan Default Prediction | Big Data - 396k Loans

## Author
**Mthunzi Ntombela**  
Johannesburg, Gauteng | 083 981 6871 | m2nzhiey@gmail.com  
GitHub: https://github.com/Mthunzy  
LinkedIn: linkedin.com/in/mthunzi-ntombela-51646b145

---

## The Problem
Banks lose millions when customers do not pay back their loans.

Can we predict who will default BEFORE giving them a loan?

This project answers that question using real LendingClub loan data.

---

## The Dataset
- **Source:** LendingClub Loans
- **Size:** 396,000 loan records - Big Data
- **Target Column:** `not.fully.paid`
    - `1` = Customer DID NOT pay back = **Default**
    - `0` = Customer DID pay back = **No Default**

---

## What I Did

1. **Loaded Big Data** - Loaded 396k rows with Pandas
2. **Explored Data** - Checked columns, data types, missing values
3. **Cleaned Data** - Removed rows with missing values
4. **Created Target** - Created `default` column from `not.fully.paid`
5. **Encoded Features** - Converted text columns like `purpose` to numbers with One-Hot Encoding
6. **Split Data** - 80% Train / 20% Test
7. **Trained Model** - Random Forest Classifier with 100 trees
8. **Evaluated Model** - Accuracy, Classification Report, Confusion Matrix
9. **Feature Importance** - Found which features cause default

---

## Key Findings

- **Interest rate is the #1 predictor of default**
- Higher interest rate = Higher risk of default
- Loan amount and credit score also strongly affect default risk
- Banks can use this model to flag high-risk applicants before approving a loan
- This can reduce bad loans and save the bank millions

---

## Tools Used
Python | Pandas | Scikit-Learn | Random Forest | Git & GitHub

---

## Project Structure
