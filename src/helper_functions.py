# Data Manipulation / Viewing Functions

def unique_data(df):
    '''
    This function takes in a dataframe and print out the unique ouputs in every column.

    Parameters
    ----------
    df : [Dataframe]
        [A dataframe with identifiable columns.]
    '''
    for i in df.columns:
        print(f'Unique outputs for column {i} :', '\n',
              df[i].unique())
        print('\n')


