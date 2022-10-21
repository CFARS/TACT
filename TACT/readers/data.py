try:
    from TACT import logger
except ImportError:
    pass
from TACT.readers.config import Config
from future.utils import itervalues, iteritems
import pandas as pd
import re
import sys
from string import printable
import numpy as np


class Data(Config):
    """
    Class to hold data, derivative features, and metadata for TACT analysis
    of a single site. Inherits from Config class

    Attributes
    ----------
    inputdata : DataFrame
        DataFrame with all anemometer, RSD data, and site atmospheric data
    timestamps : DateTime array
        array of timestamps for each row of inputdata
    a : numpy array
        Edges of Reference TI bins
    lab_a : numpy array
        Center values of Reference TI bins
    RSD_alphaFlag : bool
        whether there exists an additional upper of lower RSD height to
        compute wind shear
    Ht_1_RSD : int
        Lower height to compute wind shear from RSD
    Ht_1_RSD : int
        Upper height to compute wind shear from RSD

    """

    def get_inputdata(self):
        """Ingests and formats data from inputdata and config data

        Parameters
        ----------
        None
            Uses object attributes

        Returns
        -------
        Silent
            Sets inputdata attribute to pandas array

        """
        filename = self.input_filename
        if str(filename).split(".")[-1] == "csv":
            self.inputdata = pd.read_csv(self.input_filename)
        elif str(filename).split(".")[-1] == "xlsx":
            self.inputdata = pd.read_excel(self.input_filename)
        else:
            print(
                "Unkown input file type for the input data , please consider changing it to csv"
            )
            sys.exit()
        try:
            rename_cols = self.set_inputdataformat()
        except Exception as e:
            print("There is an error in the configuration file")
            sys.exit()

        # Look for degrees symbols delete it from input data dir columns, or any non-printable character
        ColList = self.inputdata.columns.tolist()
        for item in ColList:
            if set(item).difference(printable):
                filtered_string = "".join(filter(lambda x: x in printable, item))
                ColList = [filtered_string if x == item else x for x in ColList]
            else:
                pass
        # rename input columns to standardized columns
        self.inputdata.columns = ColList
        self.inputdata.rename(index=str, columns=rename_cols, inplace=True)
        keepCols = list(rename_cols.values())
        delCols = [x for x in self.inputdata.columns.to_list() if x not in keepCols]
        self.inputdata.drop(columns=delCols, inplace=True)

        if self.inputdata.empty == True:
            print(
                "Error no data to analyze. Inputdata dataframe is empty. Check input data."
            )
            sys.exit()
        self.timestamps = self.inputdata["Timestamp"]

        # Get Hour from timestamp and add as column
        Timestamps_dt = pd.to_datetime(self.timestamps)

        def hr_func(ts):
            h = ts.hour
            m = ts.minute / 60
            return h + m

        if "Hour" in self.inputdata.columns.to_list():
            pass
        else:
            Hour = Timestamps_dt.apply(hr_func)
            self.inputdata["Hour"] = Hour

        # drop timestamp colum from inputdata data frame, replace any 9999 cells with NaN's
        self.inputdata.drop("Timestamp", axis=1, inplace=True)
        self.inputdata.replace(9999, np.NaN, inplace=True)

        # flag any non-numeric data rows to the user
        nonNumericData_rows = self.inputdata[~self.inputdata.applymap(np.isreal).all(1)]
        if len(nonNumericData_rows) > 0:
            print(
                "Error encountered. Input data contains non numeric values, please handle this in input data before running the tool."
            )
            sys.exit()

        # make sure we have a TI column
        if "RSD_TI" in self.inputdata.columns.to_list():
            pass
        else:
            if "RSD_SD" in self.inputdata.columns.to_list():
                self.inputdata["RSD_TI"] = (
                    self.inputdata["RSD_SD"] / self.inputdata["RSD_WS"]
                )
            else:
                print(
                    "ERROR: input data does not have an RSD_TI column or an RSD_SD column. Please fix input data"
                )
                sys.exit()

        # Representative TI by bin (Representative TI = TI * 1.28 TI Std. Dev.) (Characteristic TI = TI * 1 TI Std. Dev.)

        self.inputdata["bins"] = self.inputdata["Ref_WS"].round(
            0
        )  # this acts as bin because the bin defination is between the two half integer values
        bins_p5_interval = pd.interval_range(
            start=0.25, end=20, freq=0.5, closed="left"
        )  # this is creating a interval range of .5 starting at .25
        out = pd.cut(x=self.inputdata["Ref_WS"], bins=bins_p5_interval)

        # create bin p5 category for each observation
        self.inputdata["bins_p5"] = out.apply(
            lambda x: x.mid
        )  # the middle of the interval is used as a catagorical label

        self.inputdata = self.inputdata[
            self.inputdata["Ref_TI"] != 0
        ]  # we can only analyze where the ref_TI is not 0

    def set_inputdataformat(self):
        """Formats header data from configuration file

        Converts input header data to CFARS uniform headers, and returns dict of pairs

        Parameters
        ----------
        None
            Uses object attributes

        Returns
        ------
        dict
            Dictionary of column name pairs
        """

        df = pd.read_excel(self.config_file, usecols=[0, 1]).dropna()

        df = df[
            (
                (df["Header_YourData"] != "RSD_model")
                & (df["Header_YourData"] != "height_meters")
                & [re.search("adjustment", val) is None for val in df.Header_YourData]
            )
        ]

        # Run a quick check to make sure program exits gracefully if user makes a mistake in config
        #  (i.e, not having necessary variables, duplicate variables etc.)
        intColList = df.Header_YourData.tolist()
        # look for degrees symbol and remove it
        for item in intColList:
            if set(item).difference(printable):
                filtered_string = "".join(filter(lambda x: x in printable, item))
                intColList = [filtered_string if x == item else x for x in intColList]
            else:
                pass
        cfarsColList = df.Header_CFARS_Python.tolist()

        if len(intColList) != len(set(intColList)):
            sys.exit(
                'Looks like you have duplicate variables in the "Header_YourData" portion of the table, please correct and run again'
            )

        if len(cfarsColList) != len(set(cfarsColList)):
            sys.exit(
                'Looks like you have duplicate variables in the "Header_CFARS_Python" portion of the table, please correct and run again'
            )

        # Run another quick check to ensure data fields that are necessary for analysis are entered. We MUST have reference
        requiredData = ["Ref_TI", "Ref_WS", "Ref_SD", "Timestamp"]
        if (set(requiredData).issubset(set(cfarsColList))) == False:
            missing = set(requiredData).difference(set(cfarsColList))
            sys.exit(
                "You are missing the following variables in the Header_CFARS_Python that are necessary:\n"
                + str(missing)
                + "\n Please fix and restart to run"
            )
        # Check to see if we have an RSD to compare with
        requiredData = ["RSD_TI", "Ref_TI", "RSD_WS", "Ref_WS", "Timestamp", "Ref_SD"]
        if (set(requiredData).issubset(set(cfarsColList))) == False:
            missing = set(requiredData).difference(set(cfarsColList))
            print(
                "You are unable to apply all RSD adjustment methods, skipping RSD adjustments due to missing:\n"
                + str(missing)
                + "\n Please fix in order to run adjustment methods"
            )
            requiredData = ["Ane2_TI", "Ane2_WS", "Ane2_SD"]
            if (set(requiredData).issubset(set(cfarsColList))) == False:
                missing = set(requiredData).difference(set(cfarsColList))
                sys.exit(
                    "You are missing: "
                    + str(missing)
                    + "to compare to the reference instead of RSD.\n"
                    + "\n Please fix and restart to run"
                )
        return dict(zip(intColList, cfarsColList))

    def get_refTI_bins(self):
        """Create column to group data by ref TI bins

        Parameters
        ----------
        None

        Returns
        -------
        Silent
            Adds column RefTI_bins to self.inputdata. Sets attributes self.a, self.lab_a.
        """

        self.inputdata["RefTI_bins"] = self.inputdata["Ref_TI"]
        a = np.linspace(0, 1.0, 40)
        self.a = [round(i, 3) for i in a]
        lab_a = [(a + b) / 2 for a, b in zip(a, a[1:])]
        self.lab_a = [round(a, 3) for a in lab_a]
        L_a = len(lab_a)
        self.inputdata["RefTI_bins"] = pd.cut(
            self.inputdata["RefTI_bins"], bins=L_a, labels=lab_a
        )

    def check_for_alphaConfig(self):
        """Checks to see if there exist heights to compute wind shear from RSD

        Parameters
        ----------
        None

        Returns
        -------
        Silent
            Sets Ht_1_rsd and Ht_2_rsd attributes.
        """
        self.RSD_alphaFlag = False

        # get list of available data columns and available ancillary data
        availableData = (
            pd.read_excel(self.config_file, usecols=[1], nrows=1000)
            .dropna()["Header_CFARS_Python"]
            .to_list()
        )
        if (
            "RSD_alpha_lowHeight" in availableData
            and "RSD_alpha_highHeight" in availableData
        ):
            self.RSD_alphaFlag = True
            configHtData = pd.read_excel(
                self.config_file, usecols=[3, 4], nrows=25
            ).iloc[[20, 21]]
            self.Ht_1_rsd = configHtData["Selection"].to_list()[0]
            self.Ht_2_rsd = configHtData["Selection"].to_list()[1]
        else:
            print(
                "%%%%%%%%% Warning: No alpha calculation. To compute alpha check config file settings. %%%%%%%%%%%"
            )
            self.Ht_1_rsd = None
            self.Ht_2_rsd = None
