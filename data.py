import pandas as pd

from pathlib import Path

DATA_PATH = Path(__file__).parent / 'data' / 'water-treatment.complete'
USED_COLS = ['SS-E (input suspended solids to plant)', 'SS-S (output suspended solids)', 'SED-E (input sediments to plant)', 'SED-S (output sediments)', 'PH-E (input pH to plant)', 'PH-S (output pH)', 'COND-E (input conductivity to plant)', 'COND-S (output conductivity)', 'DBO-E (input Biological demand of oxygen to plant)', 'DBO-S (output Biological demand of oxygen)', 'DQO-E (input chemical demand of oxygen to plant)', 'DQO-S (output chemical demand of oxygen)']

OUTPUT_PATH = Path(__file__).parent / 'data' / 'water-treatment.preprocessed'

def main():
    data = pd.read_csv(DATA_PATH, usecols=USED_COLS)
    generate_preprocessed_data(data, USED_COLS)
    
def generate_preprocessed_data(data, cols):
    '''
        Generate preprocessed data from given data.
        Write preprocessed data to OUTPUT_PATH
    '''
    # Before preprocessing
    print(data)

    # Preprocessing
    preprocessed_data = preprocessed(data, USED_COLS)

    # After preprocessing
    print(preprocessed_data)

    # Write preprocessed data to OUTPUT_PATH
    preprocessed_data.to_csv(OUTPUT_PATH, index=False)

def preprocessed(data, cols):
    '''
        Returns preprocessed data.
        The proprocessing methods include removing datapoint with bad value or outlier in any of the columns
    '''

    # Filter unused columns
    data = data[cols]

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

    # Normalization with min-max scaling
    min_value, max_value = data.min(), data.max()
    data = (data - min_value) / (max_value - min_value)

    # Returning pre-processed data
    return data

if __name__ == '__main__':
    main()