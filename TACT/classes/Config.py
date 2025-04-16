from legacy import logger
import pandas as pd
import os
import argparse
import re
import sys
from readers.config import get_extrap_metadata, check_for_extrapolations, get_optional_height_names

class Config(object):

    """Class to hold all Configuration and metadata associated with TACT
    analysis for a single site.

    Attributes
    ----------
    input_filename : str
        full path of input data file
    config_file : str
        full path of configuration Excel spreadsheet
    rtd_files : str
        full path of WindCube Real Time Data (RTD) files, if any
    results_file : str
        filename (new) of Excel spreadsheet with output results
    save_model_location : str
        folder to save derived adjustment model
    time_test_flag : bool
        Flag whether to test time sensitivity of model derivation
    global_model
        full path to L-TERRA or other adjustment model pickle file
    outpath_dir : str
        folder for new results_file
    outpath_file : str
        seemngly the same as results_file (?)

    site_metadata :
        TBD
    config_metadata :
        TBD
    extrapolation_type :
        TBD
    extrap_metadata :
        TBD
    available_data :
        TBD
    RSDtype :
        TBD
    adjustments_metadata :
        TBD
    """

    def __init__(
        self,
        input_filename="",
        config_file="",
        rtd_files="",
        results_file="",
        save_model_location="",
        time_test_flag="",
        global_model="",
        outpath_dir="",
        outpath_file="",
    ):

        logger.info("initiated Config object")

        self.input_filename = input_filename
        self.config_file = config_file
        self.rtd_files = rtd_files
        self.results_file = results_file
        self.save_model_location = save_model_location
        self.time_test_flag = time_test_flag
        self.global_model = global_model

        self.outpath_dir = os.path.dirname(results_file)
        self.outpath_file = os.path.basename(results_file)

        if not input_filename or not config_file or not results_file:
            self.get_input_files()

        self.model, self.height = self.get_phaseiii_metadata()
        self.apply_adjustment = self.check_for_adjustments()

    def get_input_files(self):
        """Used when running tool as script with command line arguments

        Parses arguments from command line.
        """

        logger.info(f"parsing arguments from commandline call")

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-in", "--input_filename", help="print this requires the input filename"
        )
        parser.add_argument(
            "-config",
            "--config_file",
            help="this requires the excel configuration file",
        )
        parser.add_argument(
            "-globalModel",
            "--global_model_to_test",
            help="specify the global model to test on the data",
            default="RF_model_1_SS_LTERRA_MLb.pkl",
        )
        parser.add_argument(
            "-rtd",
            "--rtd_files",
            help="this requires input directory for wincube rtd files",
            default=False,
        )
        parser.add_argument(
            "-res", "--results_file", help="this requires the excel results file"
        )
        parser.add_argument(
            "-saveModel",
            "--save_model_location",
            help="this argument specifies the location to save output global model",
            default=False,
        )
        parser.add_argument(
            "-timetestFlag",
            "--timetestFlag",
            action="store_true",
            help="initiates timing tests for model generation",
        )
        args = parser.parse_args()

        logger.info("input data file: {}".format(args.input_filename))
        logger.info("input configuration file: {}".format(args.config_file))
        logger.info("results file: {}".format(args.results_file))
        logger.info("1 HZ files directory: {}".format(args.rtd_files))
        logger.info("global model: {}".format(args.global_model_to_test))

        print("the input data file is {}".format(args.input_filename))
        print("the input configuration file is {}".format(args.config_file))
        print("results will output to {}".format(args.results_file))
        print("windcube 1 HZ rtd. files are located {}".format(args.rtd_files))
        print("Testing {} as global model".format(args.global_model_to_test))

        self.input_filename = args.input_filename
        self.config_file = args.config_file
        self.rtd_files = args.rtd_files
        self.results_file = args.results_file
        self.save_model_location = args.save_model_location
        self.time_test_flag = args.timetestFlag
        self.global_model = args.global_model_to_test

        self.outpath_dir = os.path.dirname(self.results_file)
        self.outpath_file = os.path.basename(self.results_file)

    def get_site_metadata(self):
        """
        Parameters
        ----------
        config_file: str or DataFrame
            path to input configuration file or Pandas DataFrame object

        Returns
        -------
        DataFrame
            metadata containing information about site
        """
        if isinstance(self.config_file, pd.DataFrame):
            self.site_metadata = self.config_file

        else:
            self.site_metadata = pd.read_excel(
                self.config_file, usecols=[3, 4, 5], nrows=20
            )

    def get_filtering_metadata(self):
        """Create DataFrame Metadata containing information about filtering applied to data"""

        self.config_metadata = pd.read_excel(
            self.config_file, usecols=[7, 8, 9], nrows=8
        )

    def get_adjustments_metadata(self):
        """create metadata object containing information about which adjustments will be applied
        to this data set and why
        """
        # get list of available data columns and available ancillary data
        self.available_data = (
            pd.read_excel(self.config_file, usecols=[1], nrows=1000)
            .dropna()["Header_CFARS_Python"]
            .to_list()
        )
        # check for .rtd files
        rtd = False
        mainPath = os.path.split(self.config_file)[0]
        if os.path.isdir(os.path.join(mainPath, "rtd_files")):
            rtd = True
        # read height data
        availableHtData = [s for s in self.available_data if "Ht" in s]
        configHtData = pd.read_excel(self.config_file, usecols=[3, 4], nrows=17).iloc[
            [3, 12, 13, 14, 15]
        ]
        primaryHeight = configHtData["Selection"].to_list()[0]
        # read RSD Type
        self.RSDtype = pd.read_excel(self.config_file, usecols=[4], nrows=8).iloc[6]
        # read NDA status
        ndaStatus = pd.read_excel(self.config_file, usecols=[12], nrows=3).iloc[1]
        # check argument that specifies global model
        globalModel = self.global_model
        # check ability to compute extrapolated TI
        (
            all_heights,
            ane_heights,
            RSD_heights,
            ane_cols,
            RSD_cols,
        ) = self.check_for_additional_heights(primaryHeight)
        self.extrapolation_type = check_for_extrapolations(ane_heights, RSD_heights)

        if self.extrapolation_type is not None:
            self.extrap_metadata = get_extrap_metadata(
                ane_heights, RSD_heights, self.extrapolation_type
            )

        else:
            self.extrap_metadata = pd.DataFrame(
                [
                    [
                        "extrapolation type",
                        "None",
                        "No extrapolation due to insufficient anemometer heights",
                    ]
                ],
                columns=["Type", "Height (m)", "Comparison Height Number"],
            )

        # Make dictionary of potential methods, Note: SS-LTERRA-WC-1HZ, G-LTERRA-WC-1HZ, and G-Std are windcube only (but we want to test on zx) so they are false until we know sensor
        adjustments_manager = {
            "SS-SF": True,
            "SS-S": True,
            "SS-SS": True,
            "SS-Match2": True,
            "SS-WS": True,
            "SS-WS-Std": True,
            "SS-LTERRA-WC-1HZ": False,
            "SS-LTERRA-MLa": True,
            "SS-LTERRA-MLb": True,
            "SS-LTERRA-MLc": True,
            "TI-Extrap": False,
            "G-Sa": True,
            "G-SFa": True,
            "G-Sc": True,
            "G-SFc": True,
            "G-Std": False,
            "G-Match": True,
            "G-Ref-S": True,
            "G-Ref-SF": True,
            "G-Ref-SS": True,
            "G-Ref-WS-Std": True,
        }
        # input data checking
        subset = ["Ref_TI", "RSD_TI"]
        result = all(elem in self.available_data for elem in subset)
        if result:
            pass
        else:
            if self.RSDtype["Selection"] != "No RSD":
                print(
                    "Error encountered. Input data does not, but should have TI from reference and/or TI from RSD"
                )
                sys.exit()
            else:
                subset2 = ["Ref_TI", "Ane2_TI"]
                result2 = all(elem in self.available_data for elem in subset2)
                if result2:
                    pass
                else:
                    print(
                        "Error encountered. Input data does not have enough TI data (second Anemometer) to compare. Check input and config"
                    )
                    sys.exit()

        # enable methods
        if self.RSDtype["Selection"][0:4] == "Wind":  # if rsd is windcube
            adjustments_manager["G-C"] = True
            if self.rtd_files == False:
                print(
                    "Rtd file location not specified. Not running 1Hz adjustment. To change this behavior, use argument -rtd_files"
                )
            else:
                adjustments_manager["SS-LTERRA-WC-1HZ"] = True
                adjustments_manager["G-LTERRA-WC-1HZ"] = True
        subset3 = ["RSD_SD"]
        result3 = all(elem in self.available_data for elem in subset3)
        if result3:
            pass
        else:
            print(
                "Error encountered. Input data does not include RSD standard deviation and cannot utilize some adjustments methods. Please modify input data to include standard deviation."
            )
            sys.exit()

        if self.extrapolation_type is not None:
            adjustments_manager["TI-Extrap"] = True
            adjustments_manager["Name global model"] = globalModel

        self.adjustments_metadata = adjustments_manager

    def check_for_additional_heights(self, height):
        """
        Check if columns are specified for other heights, and extract the column names.
        :param config_file: Input configuration file
        :param height: Primary comparison height (m)
        :return all_heights: dictionary of all height labels and values
        :return ane_heights: dictionary of height labels and values for anemometers
        :return RSD_heights: dictionary of height labels and values for RSD
        :return ane_cols: list of column names corresponding to additional anemometer heights
        :return RSD_cols: list of column names corresponding to additional RSD heights
        """
        # Get dictionary of all heights
        all_heights = self.get_all_heights(height)

        df = pd.read_excel(self.config_file, usecols=[0, 1]).dropna()

        # Run a quick check to make sure program exits gracefully if user makes a mistake in config
        #  (i.e, not having necessary variables, duplicate variables etc.)
        cfarsColList = df.Header_CFARS_Python.tolist()

        # Check to see if we have additional heights
        optional_height_names = get_optional_height_names()
        all_cols = []
        for cols in optional_height_names:
            col_present = [col in cfarsColList for col in cols]
            if not any(col_present):
                continue
            if (set(cols).issubset(set(cfarsColList))) == False:
                missing = set(cols).difference(set(cfarsColList))
                sys.exit(
                    "You have not specified all variables for each measurement at an additional height.\n"
                    + "You are missing the following variables in the Header_CFARS_Python that are necessary:\n"
                    + str(missing)
                    + "\n Please fix and restart to run"
                )
            all_cols.extend(cols)

        RSD_cols = [col for col in all_cols if "RSD" in col]
        ane_cols = [col for col in all_cols if "Ane" in col]

        # Get the height numbers of any Additional Comparison Heights
        regexp = re.compile("[a-zA-Z_]+(?P<ht>\d+)")
        ane_hts = [regexp.match(c) for c in ane_cols]
        ane_hts = [c.groupdict()["ht"] for c in ane_hts if c is not None]
        ane_hts = [int(h) for h in set(ane_hts)]
        RSD_hts = [regexp.match(c) for c in RSD_cols]
        RSD_hts = [c.groupdict()["ht"] for c in RSD_hts if c is not None]
        RSD_hts = [int(h) for h in set(RSD_hts)]

        # Check for Additional Comparison Heights not specified in the config file
        missing_ane = set(ane_hts).difference(set(all_heights))
        missing_RSD = set(RSD_hts).difference(set(all_heights))
        missing = missing_ane.union(missing_RSD)
        if len(missing) > 0:
            sys.exit(
                "You have not specified the Additional Comparison Height (m) for each height.\n"
                + 'You are missing the following heights in the "Selection" column that are necessary:\n\n'
                + "Additional Comparison Height "
                + str(missing)
                + "\n\n"
                + "Please fix and restart to run"
            )

        # Create dictionaries with all anemometer and RSD heights
        # NOTE : here we assume it's okay if we don't have anemometer AND RSD data for EACH additional
        #        comparison height
        ane_heights = {ht: all_heights[ht] for ht in all_heights if ht in ane_hts}
        ane_heights.update(primary=all_heights["primary"])
        RSD_heights = {ht: all_heights[ht] for ht in all_heights if ht in RSD_hts}
        RSD_heights.update(primary=all_heights["primary"])

        # Do a final check in case there are any heights that were specified but no matching columns
        missing = set(all_heights.values()).difference(
            set(RSD_heights.values()).union(set(ane_heights.values()))
        )
        if len(missing) > 0:
            sys.exit(
                "You have specified an Additional Comparison Height (m) that has no corresponding data columns.\n"
                + 'The following heights in the "Selection" column have no data columns specified:\n\n'
                + "Additional Comparison Height "
                + str(missing)
                + "\n\n"
                + "Please fix and restart to run"
            )

        return all_heights, ane_heights, RSD_heights, ane_cols, RSD_cols

    def get_all_heights(self, primary_height):
        all_heights = {"primary": primary_height}

        df = pd.read_excel(self.config_file, usecols=[3, 4]).dropna()

        ix = ["Additional Comparison Height" in h for h in df["Site Metadata"]]
        additional_heights = df.loc[ix, :]
        if len(additional_heights) > 0:
            tmp = {
                i: additional_heights["Site Metadata"].str.contains(str(i))
                for i in range(1, 5)
            }
            for ht in tmp:
                if not any(tmp[ht]):
                    continue
                all_heights[ht] = float(additional_heights["Selection"][tmp[ht].values])
        return all_heights

    def get_phaseiii_metadata(self):
        """
        :param config_file: Input configuration file
        :return: metadata containing information about model used for adjustment, height of adjustment
        """
        df = pd.read_excel(self.config_file, usecols=[3, 4]).dropna()
        try:
            model = df.Selection[df["Site Metadata"] == "RSD Type:"].values[0]
        except:
            print('No Model Listed. Model coded as "unknown"')
            model = "unknown"

        try:
            height = df.Selection[
                df["Site Metadata"] == "Primary Comparison Height (m):"
            ].values[0]
        except:
            print('No height listed. Height coded as "unknown"')
            height = "unknown"

        return model, height

    def check_for_adjustments(self):

        apply_adjustment = True

        colLabels = pd.read_excel(self.config_file, usecols=[0, 1])
        colLabels = list(colLabels.dropna()["Header_CFARS_Python"])
        rsd_cols = [s for s in colLabels if "RSD" in s]
        requiredData = ["RSD_TI", "RSD_WS"]

        if (set(requiredData).issubset(set(rsd_cols))) == False:
            apply_adjustment = False

        return apply_adjustment