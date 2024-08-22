import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

def removeOutliers(df):
    # Z-Score Normalization (Standardization)
    df["energy_sum_zscore"] = (df["energy_sum"] - df["energy_sum"].mean()) / df["energy_sum"].std()
    df["curve_std_zscore"] = (df["curve_std"] - df["curve_std"].mean()) / df["curve_std"].std()
    df["time_to_peak_zscore"] = (df["time_to_peak"] - df["time_to_peak"].mean()) / df["time_to_peak"].std()

    #Removing Outliers by adding a Z-score Threshold
    return df.loc[(df["energy_sum_zscore"] <= 1.5) & (df["energy_sum_zscore"] >= -1.5) & (df["curve_std_zscore"] <= 1.5) & (df["curve_std_zscore"] >= -1.5) & (df["time_to_peak_zscore"] <= 1.5) & (df["time_to_peak_zscore"] >= -1.5)]

def cleanForCurve(csv):
    #Importing csv and dropping NaN column (RP3 has an empty ID column)
    df = pd.read_csv("curve_analysis/"+csv).drop("id", axis=1)

    #removing any strokes with a value of 0 somewhere (excluding curve data array) to remove any incorrectly recorded strokes
    df = df.dropna().loc[((df!=0).any(axis=1)) & ((df["pulse"] != 0))]

    #getting rid of any workouts that do not have pulse columns
    if (df.empty == False):
        #fix format of curve data to make it usable
        df["curve_data"] = df["curve_data"].str.split(pat = ",").map(lambda x: np.array([0]+x+[0], float))

        #Calculate More Curve Metrics
        df["curve_std"] = df["curve_data"].map(lambda x: np.std(x))
        df["time_to_peak"] = (df["peak_force_pos"]/df["stroke_length"] * df["drive_time"])

        #Add TimeStamp of date to every stroke in each workout so can potentially do timeseries analysis
        df["datestamp"] = pd.to_datetime(csv[:8])
        
        # 0 == UT2 and 1 == AT
        df["workout_type"] = 1 if (df["pulse"].mean() > 170) else 0  


        return df

data = []
for file in os.listdir("curve_analysis"):
    if file.endswith(".csv"):
        data.append(cleanForCurve(file))
df = pd.concat(data)

#Should remove remove inconsistent strokes such as breaks between intervals or if I ever stopped mid workout for whatever reason

#Removing Incorrect data (Ranges ended up inclusing strokes from beginning of a few recent AT workouts)
df_ut2 = removeOutliers(df.loc[(df["workout_type"] == 0)]).iloc[:3130, :]

df_at = removeOutliers(df.loc[(df["workout_type"] == 1)])

for i,c in enumerate(df_ut2.columns):
    print("index:" + str(i) + " name: " + c )

#Formula for AT workout Curve Score
# F/F_max * 0.2 + A/A_max * 0.2 + D/D_max * 0.1 + P/P_max * 0.3 + SL/SL_max * 0.2
w1 = 0.2
w2 = 0.2
w3 = 0.1
w4 = 0.3
w5 = 0.2

df_at["combined_curve_score"] = (
    df_at["peak_force"] / df_at["peak_force"].max() * w1 +
    df_at["energy_sum"] / df_at["energy_sum"].max() * w2 +
    df_at["distance_per_stroke"] / df_at["distance_per_stroke"].max() * w3 +
    df_at["power"] / df_at["power"].max() * w4 +
    df_at["stroke_length"] / df_at["stroke_length"].max() * w5
)

#Formula for UT2 workout Curve Score
# F/F_max * 0.3 + A/A_max * 0.3 + D/D_max * 0.1 + SL/SL_max * 0.3
w1_ut2 = 0.1#0.3
w2_ut2 = 0.4#0.3
w3_ut2 = 0.1#0.1
w4_ut2 = 0.5#0.3

df_ut2["combined_curve_score"] = (
    df_ut2["peak_force"] / df_ut2["peak_force"].max() * w1_ut2 +
    df_ut2["energy_sum"] / df_ut2["energy_sum"].max() * w2_ut2 +
    #df_ut2["distance_per_stroke"] / df_ut2["distance_per_stroke"].max() * w3_ut2 +
    df_ut2["stroke_length"] / df_ut2["stroke_length"].max() * w4_ut2
)
#Ideal stroke from AT that is above UT2 threshold as can include one of initial few strokes to get the erg to speed
max_for_at_step1 = df_at.loc[(df_at["pulse"] > 160)]
max_for_at_step2 =  max_for_at_step1.loc[(max_for_at_step1["combined_curve_score"] == max_for_at_step1["combined_curve_score"].max())]
max_for_at = max_for_at_step2["curve_data"].to_numpy()[0]

#Ideal stroke from UT2 that is within UT2 threshold as can include one of initial few strokes to get the erg to speed
max_for_ut2_step1 = df_ut2.loc[(df_ut2["pulse"] > 145) & (df_ut2["pulse"] < 160) & (df_ut2["stroke_rate"] > 15) & (df_ut2["stroke_rate"] < 23)]
max_for_ut2_step2 = max_for_ut2_step1.loc[(max_for_ut2_step1["combined_curve_score"] == max_for_ut2_step1["combined_curve_score"].max())]
max_for_ut2 = max_for_ut2_step2["curve_data"].to_numpy()[0]

print("AT Pulse at best curve:", max_for_at_step2["pulse"].to_numpy()[0])
time500m = max_for_at_step2["estimated_500m_time"].to_numpy()[0]
print("estimated_500m_time: ", f"{int(time500m//60)}:{(time500m/60 - time500m//60) * 60}")
print("Stroke Length: ", max_for_at_step2["stroke_length"].to_numpy()[0])
print("Stroke Rate: ", max_for_at_step2["stroke_rate"].to_numpy()[0])
print("Date: ", max_for_at_step2["datestamp"].to_numpy()[0])
print("Curve Score: ", max_for_at_step2["combined_curve_score"].to_numpy()[0])

print("UT2 Pulse at best curve:", max_for_ut2_step2["pulse"].to_numpy()[0])
time500m = max_for_ut2_step2["estimated_500m_time"].to_numpy()[0]
print("estimated_500m_time: ", f"{int(time500m//60)}:{(time500m/60 - time500m//60) * 60}")
print("Stroke Length: ", max_for_ut2_step2["stroke_length"].to_numpy()[0])
print("Stroke Rate: ", max_for_ut2_step2["stroke_rate"].to_numpy()[0])
print("Date: ", max_for_ut2_step2["datestamp"].to_numpy()[0])
print("Curve Score: ", max_for_ut2_step2["combined_curve_score"].to_numpy()[0])

#Plotting Ideal stroke for UT2 and AT Workouts
fig, curve = plt.subplots()

at_linspace = np.linspace(0, max_for_at_step2["stroke_length"].to_numpy()[0], len(max_for_at))
ut2_linspace = np.linspace(0, max_for_ut2_step2["stroke_length"].to_numpy()[0], len(max_for_ut2))

curve.plot(at_linspace, max_for_at, c="blue")
curve.plot(ut2_linspace, max_for_ut2, c="orange")

#Calculating Best stroke curve
best_peak_force_pos = max_for_ut2_step2["peak_force_pos"].to_numpy()[0]
best_peak_force = max_for_at_step2["peak_force"].to_numpy()[0]
best_stroke_length = max_for_ut2_step2["stroke_length"].to_numpy()[0]

spline = CubicHermiteSpline([0, best_peak_force_pos, best_stroke_length], [0, best_peak_force, 0], [8, 0, -3])
x = np.linspace(0, best_stroke_length, 70)
best_curve_data = spline(x)

curve.plot(x, best_curve_data, c="green")

plt.show()

fig2, score = plt.subplots()
score.plot(range(df_at.shape[0]), df_at["combined_curve_score"].to_numpy())
score.plot(range(df_ut2.shape[0]), df_ut2["combined_curve_score"].to_numpy())
plt.show()