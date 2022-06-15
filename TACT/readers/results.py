import pandas as pd

class Results():
    """
    Object created from TACT results .xslx file. 

    Attributes
    ----------
    results_filepath : string (path-like)
        Path to results .xslx file
    results : dict
        Dictionary of pandas dataframes of TI tables from results file
        Keys correspond to different TI adjustment methods
        DF rows for wind speed bins and columns for TI stats

    Methods
    -------
    results_from_xls 
    
    """

    def __init__(self, results_filepath):

        self.results_filepath = results_filepath
        self.results = self.results_from_xls()
    
    def results_from_xls(self, rows_to_skip=36, rows_to_keep=14):
        """
        Reads in Excel file of TACT output and returns dict of dataframes of results from each method,
        binned by 1 m/s mean wind speed.

        Parameters
        ----------
        rows_to_skip : int, default 36
            Number of rows in Excel sheets before results table begins
        rows_to_keep : int, default 14
            Number of rows to read in results table

        Returns
        -------
        results_dict : dict

        """
                
        # get sheet names from excel workbook
        with pd.ExcelFile(self.results_filepath) as xls:
            sheet_list = xls.sheet_names

        results_sheets = [sheet for sheet in sheet_list if 'SS-' in sheet or 'G-' in sheet]
        
        # read results table from each sheet (each method) into df
        xl_df_dict = pd.read_excel(self.results_filepath, results_sheets, skiprows=rows_to_skip, nrows=rows_to_keep)
        
        #create a dict of reformatted dataframes with results from each method
        results_dict = {}

        for method in xl_df_dict:
            results_df = xl_df_dict[method].iloc[1:][[col for col in xl_df_dict[method].columns if 'mean' in col]].T
            results_df.columns = xl_df_dict[method].iloc[1:,0]
            results_df['ws_bin'] = [float(idx[5:]) for idx in results_df.index]
            results_dict[method] = results_df
        
        return results_dict