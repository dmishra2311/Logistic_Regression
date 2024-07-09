import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_excel("Tobacco Screening Patient.xlsx")

#Display basic information
print(df.info())
print(df.describe())
print(df.head())

lung_cancer_history_count = df['LungCancerHistory'].sum()
print(f"Number of entries where LungCancerHistory is 1: {lung_cancer_history_count}")

#Print missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

#Plot missing values
plt.figure(figsize=(10, 10))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()

#Set ZIP as a string
df['ZIP'] = df['ZIP'].astype(str)

#Find entries with ZIP codes less than 5 digits
zip_less_than_5_digits = df[df['ZIP'].apply(lambda x: len(x) < 5)]

#Find entries with ZIP codes greater than 5 digits
zip_greater_than_5_digits = df[df['ZIP'].apply(lambda x: len(x) > 5)]

#Count the entries
count_less_than_5_digits = zip_less_than_5_digits.shape[0]
count_greater_than_5_digits = zip_greater_than_5_digits.shape[0]

print(f"Number of entries with ZIP codes less than 5 digits: {count_less_than_5_digits}")
print(f"Number of entries with ZIP codes greater than 5 digits: {count_greater_than_5_digits}")

#Replace all blanks with NaN/NaT
df.replace("", pd.NA, inplace=True)

#Drop PatUniqueID and PCP_ID columns
df.drop(columns=['PatUniqueID', 'PCP_ID'], inplace=True)

#Convert all ZIP codes to 5 digit
df['ZIP'] = df['ZIP'].astype(str)
df['ZIP'] = df['ZIP'].apply(lambda x: x.zfill(5) if len(x) == 5 else x[:5])

#Handle ZIP codes less than 5 digits by dropping
df = df[df['ZIP'].apply(lambda x: len(x) == 5)]

#Convert categorical variables to numeric using one-hot encoding
#setting drop_first=True to avoid multicollinearity
df = pd.get_dummies(df, columns=['TobaccoUse', 'Sex', 'Race', 'Ethnicity'])
boolean_columns = df.select_dtypes(include=['bool']).columns
df[boolean_columns] = df[boolean_columns].astype(int)

#Convert date fields to datetime format
df['QuitDT'] = pd.to_datetime(df['QuitDT'], errors='coerce')
df['Latest_LDCT'] = pd.to_datetime(df['Latest_LDCT'], errors='coerce')

#Calculate QuitYrs if QuitDT is available and QuitYrs is missing
current_year = datetime.now().year
df['QuitYrs'] = df.apply(
    lambda row: current_year - row['QuitDT'].year if pd.notna(row['QuitDT']) and 
    pd.isna(row['QuitYrs']) else row['QuitYrs'], axis=1
)
#Drop QuitDT column
df.drop(columns=['QuitDT'], inplace=True)

#Drop TobaccoUse_Smoker, Current Status Unknown
#there are only 6 records and none have lung cancer history
df.drop(columns=['TobaccoUse_Smoker, Current Status Unknown'], inplace=True)

#Create a binary indicator for existing Latest_LDCT
df['Latest_LDCT_Exists'] = df['Latest_LDCT'].notna().astype(int)
df.drop(columns=['Latest_LDCT'], inplace=True)

#Combine Race_XXX fields into Race_Other except for Race_White and Race_Black 
#or African American
race_columns = [col for col in df.columns if col.startswith('Race_') 
                and col not in ['Race_White', 'Race_Black or African American', 
                                'Race_Other']]
df['Race_Other'] = df[race_columns].sum(axis=1)
df.drop(columns=race_columns, inplace=True)

#Combine Ethnicity_XXX fields into Ethnicity_Hispanic, Ethnicity_Non-Hispanic, 
#Ethnicity_Other
df['Ethnicity_Non-Hispanic'] = df['Ethnicity_Not Hispanic, Latino/a, or Spanish origin']
df['Ethnicity_Unknown'] = df[['Ethnicity_Decline to Answer', 'Ethnicity_Unknown']
                             ].sum(axis=1)
df['Ethnicity_Hispanic'] = df[['Ethnicity_Cuban', 'Ethnicity_Mexican, Mexican American, or Chicano/a', 
                               'Ethnicity_Other Hispanic, Latino/a, or Spanish origin', 'Ethnicity_Puerto Rican']
                               ].sum(axis=1)

# Drop the original Ethnicity columns
ethnicity_columns = ['Ethnicity_Cuban', 'Ethnicity_Decline to Answer', 
                     'Ethnicity_Mexican, Mexican American, or Chicano/a', 
                     'Ethnicity_Not Hispanic, Latino/a, or Spanish origin', 
                     'Ethnicity_Other Hispanic, Latino/a, or Spanish origin', 
                     'Ethnicity_Puerto Rican']
df.drop(columns=ethnicity_columns, inplace=True)

#Drop columns with all zero values
df = df.loc[:, (df != 0).any(axis=0)]

#Boolean fields calculations
boolean_fields = [col for col in df.columns if set(df[col].dropna().unique()) <= {0, 1}]
one_counts = df[boolean_fields].sum()
print("Count of 1 values in boolean fields:\n", one_counts)

#Non-null counts for all other fields
non_null_counts = df.drop(columns=boolean_fields).notna().sum()
print("Count of non-null values in other fields:\n", non_null_counts)

#Export DataFrame to Excel
df.to_excel('Cleaned UHS Data.xlsx', index=False)

# one_counts_less_than_105 = one_counts[one_counts < 105]
# print("Boolean columns with count of 1 values less than 105:\n", one_counts_less_than_105)

# # Plot count of 1 values for boolean columns with less than 105 occurrences
# plt.figure(figsize=(15, 5))  # Set the figure size before plotting
# sns.barplot(x=one_counts_less_than_105.index, y=one_counts_less_than_105.values)
# plt.title('Count of 1 Values in Boolean Fields with Less Than 105 Occurrences')
# plt.xticks(rotation=45, ha='right', fontsize=8)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.3)
# plt.show()