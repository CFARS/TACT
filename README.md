# ![CFARS](assets/cfars_logo_48.png) TACT site_suitability_tool

TACT = Turbulence intensity Adjustment Comparison Tool

1. clone or download repository 
2. create separate .csv file for each data set 
3. Fill out a separate configuration template file for each project data set 
4. Execute TACT.py
``` bash
python TACT.py -in PATH_TO_DATA.csv -config PATH_TO_CONFIG.xlsx -res PATH_TO_RESULTS_FILE.xlsx --timetestFlag
```
5. Send results output and configuration files to aea@nrgsystems.com 

*** contact aea@nrgsystems.com for assistance


Updated 3/25/2025 by CJP

To run existing legacy TACT implementation, execute from `/legacy`:
`python3 TACT.py -in Example/example_project.csv -config Example/configuration_example_project.xlsx -res Example/out_example_project.xlsx --timetestFlag`

To run the revised TACT implementation, execute `python3 main.py` from within `/tact`