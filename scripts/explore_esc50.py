import pandas as pd

df=pd.read_csv('ESC-50/meta/esc50.csv')
print(df.head()) 
print(df.info()) #prints informations like data types, non-null values, memory usage etc.
print(df.shape)  #prints number of rows and columns in dataset
print(df.columns) 

print(df['category'].unique()) #prints unique categories in dataset


label_map = {
    'coughing': 'cough',
    'sneezing': 'sneeze',
    'snoring': 'snore',
    'breathing': 'background',
    'vacuum_cleaner': 'background',
    'rain': 'background',
    'wind': 'background',
    'clock_tick': 'background',
    'laughing': 'background',
    'pouring_water': 'background',
    'clapping': 'background',
}

filtered_df = df[df['category'].isin(label_map.keys())].copy() #filters rows where category is in the label_map keys

filtered_df['label'] = filtered_df['category'].map(label_map) #maps the category to new label using label_map

print(filtered_df.shape)  #prints number of rows and columns in filtered dataset

print(filtered_df['label'].value_counts()) #prints count of each label in filtered dataset

filtered_df['filepath'] = "ESC-50/audio/" + filtered_df['filename'] #creates a new column 'filepath' by concatenating the base path with the filename

print(filtered_df[['filepath', 'label']].head())
