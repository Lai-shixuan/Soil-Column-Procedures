# %%
import pandas as pd
import sqlite3

# %%
csv_file_path = 'f:/3.Experimental_Data/Soils/Scan-parameters.csv'
df = pd.read_csv(csv_file_path, na_values=['NA', 'null'])
print(df.head())

# Handle missing values as needed
# For example, you can fill NaNs with a default value or drop rows with missing values
# Here, we'll fill NaNs with None to be compatible with SQLite
df = df.where(pd.notnull(df), None)
print(df.head())

# %%

def extract_first_number(value):
    if pd.isna(value) or value == 'none':
        return 0
    try:
        return int(str(value).split('-')[0])
    except ValueError:
        return 0

df['Physical_size_um'] = df['Straightened_img_size'].apply(extract_first_number) * df['Resolution_um'].apply(extract_first_number)

print(df.head())

# %%
# Connect to the SQLite database (or create it if it doesn't exist)
# Replace 'metadata.db' with your desired database name
conn = sqlite3.connect('metadata.db')

# Specify the name of the table where the data will be stored
table_name = 'metadata_table'

# Write the DataFrame to the SQLite database
df.to_sql(table_name, conn, if_exists='replace', index=False)

print(f"\nData has been successfully written to the '{table_name}' table in 'metadata.db' database.")

# Close the database connection
conn.close()
