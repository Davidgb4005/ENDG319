import pandas as pd
def ParseCar1():
    input_file = 'Car1.csv'
    output_file = 'CarParsed.csv'
    data = pd.read_csv(input_file)
    data.columns = data.columns.str.strip()
    columns_to_drop = ['Make', 'Model', 'Country']
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    class_mapping = {'Europe': 1, 'Asia': 2, 'North America': 3}
    data['Class'] = data['Class'].str.strip().map(class_mapping)
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col}: {e}")
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col].fillna(data[col].mean(), inplace=True)
    data.to_csv(output_file, index=False)
