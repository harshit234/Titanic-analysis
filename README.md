📌 Project Overview

This project performs an end-to-end analysis of the Titanic dataset to identify key factors influencing passenger survival. The analysis includes NumPy-based vectorized computations, feature engineering, survival insights, and a custom weighted prediction model.

The goal is to extract business-level insights while demonstrating strong data manipulation and modeling fundamentals without relying on Pandas groupby.

🎯 Objectives

Analyze survival patterns by Gender and Passenger Class

Identify key survival drivers

Build a fully vectorized NumPy survival scoring system

Create a custom predict_survival() function

Generate business-oriented recommendations

📊 Key Findings

Women survival rate ≈ 75%

First-class passengers had significantly higher survival

Children were prioritized during evacuation

Survival importance ranking:

Gender > Class > Fare > Age > Family Size

🔥 Bonus Challenge (Advanced)

Implemented using NumPy only (no Pandas groupby, no loops):

✔ Survival by Gender (Vectorized)
✔ Survival by Class (Vectorized)
✔ Fully vectorized weighted survival score
✔ Custom prediction function:

predict_survival(passenger_dict)

This demonstrates:

Boolean masking

NumPy vectorization

Feature engineering

Custom inference logic

Clean mathematical modeling

🧠 Model Explanation

A weighted scoring system was used to estimate survival probability:

Score =
0.35 * Gender +
0.25 * Class +
0.15 * Fare +
0.15 * AgeGroup +
0.10 * FamilySize

Passengers with score > 0.5 are predicted to survive.
