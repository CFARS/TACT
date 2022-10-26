# ---                ExampleDataTest.ps1
#
# For local testing on Windows computers
#
# This is not intended to be used in pipelines or as a replacement
# for unit tests

$inFile = ".\Example\example_project.csv"
$config = ".\Example\configuration_example_project.xlsx"
$outFile = ".\test\test_output.xlsx"
# $tiOutFile = ".\test\TI_10minuteAdjusted_test_output.csv"

.\venv\Scripts\Activate.ps1

pip install -e .

python TACT.py -in $inFile -config $config -res $outFile --timetestFlag

