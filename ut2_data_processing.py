import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################################################
#                  PUT ON HOLD TO GET BETTER DATA              #
################################################################

def cleanForUT2(csv):
    #Importing csv and dropping NaN column (RP3 has an empty ID column)
    df = pd.read_csv("ut2/"+csv).drop("id", axis=1)

    #Remove any strokes outside of widely accepted range for UT2 stroke rate
    df = df[(df["stroke_rate"] > 16) & (df["stroke_rate"] < 23)]

    #fix format of curve data to make it usable ::::: col index is 21
    df["curve_data"] = df["curve_data"].str.split(pat = ",").map(lambda x: np.array([0]+x+[0], float))

    #Add TimeStamp of date to every stroke in each workout so can do timeseries analysis (maybe)
    df["datestamp"] = pd.to_datetime(csv[:8])

    return df

data = []
for file in os.listdir("ut2"):
    if file.endswith(".csv"):
        data.append(cleanForUT2(file))
df = pd.concat(data)
df = data[1]

#for i,c in enumerate(df.columns):
#    print("index:" + str(i) + " name: " + c )

x = df["pulse"]
y = df["work_per_pulse"]

plt.plot(range(len(x)), x)
plt.show()




