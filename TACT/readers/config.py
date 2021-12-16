try:
    from TACT import logger
except ImportError:
    pass


def get_inputfiles():
    """ Used when running tool as script with command line arguments
    
    Parses arguments from command line. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-in","--input_filename", help="print this requires the input filename")
    parser.add_argument("-config","--config_file", help="this requires the excel configuration file")
    parser.add_argument("-globalModel", "--global_model_to_test", help="specify the global model to test on the data",default='RF_model_1_SS_LTERRA_MLb.pkl')
    parser.add_argument("-rtd","--rtd_files", help="this requires input directory for wincube rtd files",default=False)
    parser.add_argument("-res","--results_file", help="this requires the excel results file")
    parser.add_argument("-saveModel", "--save_model_location", help="this argument specifies the location to save output global model", default=False)
    parser.add_argument("-timetestFlag", "--timetestFlag", action="store_true",
                        help = "initiates timing tests for model generation")
    args = parser.parse_args()

    logger.info('input data file: {}'.format(args.input_filename))
    logger.info('input configuration file: {}'.format(args.config_file))
    logger.info('results file: {}'.format(args.results_file))
    logger.info('1 HZ files directory: {}'.format(args.rtd_files))
    logger.info('global model: {}'.format(args.global_model_to_test))

    print('the input data file is {}'.format(args.input_filename))
    print('the input configuration file is {}'.format(args.config_file))
    print('results will output to {}'.format(args.results_file))
    print('windcube 1 HZ rtd. files are located {}'.format(args.rtd_files))
    print('Testing {} as global model'.format(args.global_model_to_test))

    return args.input_filename, args.config_file, args.rtd_files, args.results_file, args.save_model_location, args.timetestFlag, args.global_model_to_test