Bonus Challenge (Very Difficult)
"""

import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

"""Part 1-a"""

survived = df["Survived"].values
sex = df["Sex"].values

sex_encoded = np.where(sex == "female", 1, 0)

female_mask = sex_encoded == 1
male_mask = sex_encoded == 0

female_survival = survived[female_mask].mean()
male_survival = survived[male_mask].mean()

print("Female Survival Rate:", female_survival)
print("Male Survival Rate:", male_survival)

"""Part 1-b"""

pclass = df["Pclass"].values

class_1_survival = survived[pclass == 1].mean()
class_2_survival = survived[pclass == 2].mean()
class_3_survival = survived[pclass == 3].mean()

print("Class 1 Survival:", class_1_survival)
print("Class 2 Survival:", class_2_survival)
print("Class 3 Survival:", class_3_survival)

unique_classes = np.unique(pclass)

survival_by_class = {
    cls: survived[pclass == cls].mean()
    for cls in unique_classes
}

print(survival_by_class)

"""Part 2"""

df["Sex"] = np.where(df["Sex"] == "female", 1, 0)
df["AgeGroup"] = np.where(df["Age"] < 15, 1, 0)
df["FamilySize"] = df["SibSp"] + df["Parch"]

score = (
    0.35 * df["Sex"].values +
    0.25 * (1 - df["Pclass"].values / 3) +
    0.15 * df["Fare"].values / df["Fare"].max() +
    0.15 * df["AgeGroup"].values +
    0.10 * df["FamilySize"].values / df["FamilySize"].max()
)

"""Part 3"""

def predict_survival(passenger):
    """
    Predict survival using weighted scoring model.
    passenger: dictionary with keys:
    Sex, Pclass, Fare, Age, SibSp, Parch
    """

    # Encode gender
    sex = 1 if passenger["Sex"] == "female" else 0

    # Normalize class (1 best → higher score)
    class_score = 1 - (passenger["Pclass"] / 3)

    # Age group
    age_group = 1 if passenger["Age"] < 15 else 0

    # Family size
    family_size = passenger["SibSp"] + passenger["Parch"]

    # Normalize values (assume max fare 512, max family 10 approx)
    fare_norm = passenger["Fare"] / 512
    family_norm = family_size / 10

    score = (
        0.35 * sex +
        0.25 * class_score +
        0.15 * fare_norm +
        0.15 * age_group +
        0.10 * family_norm
    )

    return 1 if score > 0.5 else 0

new_passenger = {
    "Sex": "female",
    "Pclass": 1,
    "Fare": 100,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0
}

prediction = predict_survival(new_passenger)
print("Predicted Survival:", prediction)

