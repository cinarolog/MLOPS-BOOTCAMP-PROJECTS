import numpy as np
import pandas as pd

def extract_whiskers(data, whisker=1.5):
    median_value = np.median(data) # Medyan
    upper_quartile = np.percentile(data, 75) # 75%
    lower_quartile = np.percentile(data, 25) # 25% 

    iqr = upper_quartile - lower_quartile # Interquartile Range
    
    upper_whisker = data[data<=upper_quartile+whisker*iqr].max() # Maksimum Kabul Edilen Değer
    lower_whisker = data[data>=lower_quartile-whisker*iqr].min() # Minimum Kabul Edilen Değer
    
    print("Upper Whisker:", upper_whisker)
    print("Lower Whisker:", lower_whisker)

    return lower_whisker,upper_whisker