import numpy as np
import pandas as pd

my_data = np.random.rand(5, 4)
my_pd = pd.DataFrame(my_data)

my_pd.columns = ['IOU', 'F1', 'Precision', 'Recall']
my_pd.index = ['Model1', 'Model2', 'Model3', 'Model4', 'Model5']

# Fix path format and add error handling
path = r'/mnt/c/Users/laish/GoogleDrive/04 microCT土壤/20250102/traditional_model_compare.csv'
try:
    my_pd.to_csv(path)
    print(f"File successfully saved to: {path}")
except Exception as e:
    print(f"Error saving file: {e}")
    # Alternative: Save to current directory if original path fails
    my_pd.to_csv('traditional_model_compare.csv')
    print("File saved to current directory instead")