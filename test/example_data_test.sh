#!/bin/bash

INFILE=Example/example_project.csv
CONFIG=Example/configuration_example_project.xlsx
OUTFILE=test/test_output.xlsx
TIOUTFILE=test/TI_10minuteAdjusted_test_output.csv

source venv/bin/activate
pip install -e .
python3 TACT.py -in $INFILE -config $CONFIG -res $OUTFILE --timetestFlag
deactivate

# check outputs

if [ "$1" == "keep" ];
then
    REMOVEFILES=0
else
    REMOVEFILES=1
fi

if [ $REMOVEFILES == 1 ]; 
then
    if [ -f $OUTFILE ];
    then
        rm $OUTFILE
    else
        printf "ERROR: $OUTFILE not created! Exiting\n"
        exit 1
    fi

    if [ -f $TIOUTFILE ];
    then    
        rm $TIOUTFILE
    else 
        printf "ERROR: $TIOUTFILE not created! Exiting\n"
        exit 2
    fi

    printf "INFO: example_data_test passed!\n"

else
    if [ -f $OUTFILE ];
    then
        printf "$OUTFILE created\n"
    else
        printf "$OUTFILE NOT created\n"
    fi

    if [ -f $TIOUTFILE ];
    then    
        printf "$TIOUTFILE created"
    else
        printf "$TIOUTFILE created"
    fi
fi
