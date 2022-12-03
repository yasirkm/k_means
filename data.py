import pandas as pd

from pathlib import Path

DATA_PATH = Path(__file__).parent / 'data' / 'water-treatment.complete'
USED_COLS = ['Q-E (input flow to plant)', 'SS-E (input suspended solids to plant)', 'SS-S (output suspended solids)', 'SS-P (input suspended solids to primary settler)', 'SS-D (input suspended solids to secondary settler)', 'RD-SS-P (performance input suspended solids to primary settler)', 'RD-SS-G (global performance input suspended solids)']

OUTPUT_PATH = Path(__file__).parent / 'data' / 'water-treatment.preprocessed'

def main():
    data = pd.read_csv(DATA_PATH, usecols=USED_COLS)
    generate_preprocessed_data(data, USED_COLS)
    
def generate_preprocessed_data(data, cols):
    print(data)
    preprocessed_data = preprocessed(data, USED_COLS)
    print(preprocessed_data)
    preprocessed_data.to_csv(OUTPUT_PATH, index=False)

def preprocessed(data, cols):
    '''
        Returns preprocessed data.
        The proprocessing methods include removing datapoint with bad value or outlier in any of the columns
    '''

    # Removing non numeric value
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data.astype('float64')

    # Removing outlier
    q1 = data[cols].quantile(0.25)
    q3 = data[cols].quantile(0.75)
    iqr = q3-q1
    condition = ~((data[cols] < (q1 - 1.5 * iqr)) | (data[cols] > (q3 + 1.5 * iqr))).any(axis=1)
    data = data[condition]

    # Min-Max Scaling
    min_value, max_value = data.min(), data.max()
    print(min_value, max_value)
    data = (data - min_value) / (max_value - min_value)

    # Returning pre-processed data
    return data

if __name__ == '__main__':
    main()