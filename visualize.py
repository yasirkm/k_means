import matplotlib.pyplot as plt
import pandas as pd

from data import preprocessed, DATA_PATH, USED_COLS

def main():
    show_graphs()
    show_graphs(preprocess=True)

def show_graphs(data_path=DATA_PATH, cols=USED_COLS, preprocess=False):
    '''
        Shows 2D and 3D graph of audit_risk data
    '''
    # Reading data file
    data = pd.read_csv(data_path)
    if preprocess:
        data = preprocessed(data, cols)
    print(data)
    
    # Creating figure for visualization
    fig, axs = plt.subplots(2,3)
    fig.suptitle("Scatter Plots for Inputs and Outputs of Water-Treatment Plant")

    # Scatterplot for SS-E (input suspended solids to plant), and SS-S (output suspended solids)
    axs[0][0].scatter(data['SS-E (input suspended solids to plant)'], data['SS-S (output suspended solids)'], color='black', alpha=0.1, label = 'Suspended Solids')
    axs[0][0].set_ylabel('SS-S (output suspended solids)')
    axs[0][0].set_xlabel('SS-E (input suspended solids to plant)')
    axs[0][0].legend(loc='upper right')

    # Scatterplot for PH-E (input pH to plant), and PH-S (output pH)
    axs[0][1].scatter(data['PH-E (input pH to plant)'], data['PH-S (output pH)'], color='blue', alpha=0.1, label = 'PH')
    axs[0][1].set_ylabel('PH-S (output pH)')
    axs[0][1].set_xlabel('PH-E (input pH to plant)')
    axs[0][1].legend(loc='upper right')

    # Scatterplot for COND-E (input conductivity to plant), and COND-S (output conductivity)
    axs[0][2].scatter(data['COND-E (input conductivity to plant)'], data['COND-S (output conductivity)'], color='magenta', alpha=0.1, label='Conductivity')
    axs[0][2].set_ylabel('COND-S (output conductivity)')
    axs[0][2].set_xlabel('COND-E (input conductivity to plant)')
    axs[0][2].legend(loc='upper right')

    # Scatterplot for DQO-E (input chemical demand of oxygen to plant), and DQO-S (output chemical demand of oxygen)
    axs[1][0].scatter(data['DQO-E (input chemical demand of oxygen to plant)'], data['DQO-S (output chemical demand of oxygen)'], color='red', alpha=0.1, label='Chemical Demand of Oxygen')
    axs[1][0].set_ylabel('DQO-S (output chemical demand of oxygen)')
    axs[1][0].set_xlabel('DQO-E (input chemical demand of oxygen to plant)')
    axs[1][0].legend(loc='upper right')

    # Scatterplot for SED-E (input sediments to plant), and SED-S (output sediments)
    axs[1][1].scatter(data['SED-E (input sediments to plant)'], data['SED-S (output sediments)'], color='orange', alpha=0.1, label='Sediments')
    axs[1][1].set_ylabel('SED-S (output sediments)')
    axs[1][1].set_xlabel('SED-E (input sediments to plant)')
    axs[1][1].legend(loc='upper right')

    # Scatterplot for DBO-E (input Biological demand of oxygen to plant), and DBO-S (output Biological demand of oxygen)
    axs[1][2].scatter(data['DBO-E (input Biological demand of oxygen to plant)'], data['DBO-S (output Biological demand of oxygen)'], color='lime', alpha=0.1, label= 'Biological Demand of Oxygen')
    axs[1][2].set_ylabel('DBO-S (output Biological demand of oxygen)')
    axs[1][2].set_xlabel('DBO-E (input Biological demand of oxygen to plant)')
    axs[1][2].legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()