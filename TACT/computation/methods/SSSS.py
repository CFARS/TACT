import numpy as np
import pandas as pd


def perform_SS_SS_adjustment(inputdata, All_class_data, primary_idx):
    """
    simple site specific adjustment, but adjust each TKE class differently
    """
    results = pd.DataFrame(
        columns=[
            "sensor",
            "height",
            "adjustment",
            "m",
            "c",
            "rsquared",
            "difference",
            "mse",
            "rmse",
        ]
    )
    _adjuster_SS_SS = Adjustments()

    className = 1
    items_adjected = []
    for item in All_class_data:
        temp = item[primary_idx]
        if temp.empty:
            pass
        else:
            inputdata_test = temp[temp["split"] == True].copy()
            inputdata_train = temp[temp["split"] == False].copy()
            if (
                inputdata_test.empty
                or len(inputdata_test) < 2
                or inputdata_train.empty
                or len(inputdata_train) < 2
            ):
                pass
                items_adjected.append(inputdata_test)
            else:
                # get te adjustment for this TKE class
                full = pd.DataFrame()
                full["Ref_TI"] = inputdata_test["Ref_TI"]
                full["RSD_TI"] = inputdata_test["RSD_TI"]
                full = full.dropna()
                if len(full) < 2:
                    pass
                else:
                    model = _adjuster_SS_SS.get_regression(
                        inputdata_train["RSD_TI"], inputdata_train["Ref_TI"]
                    )
                    m = model[0]
                    c = model[1]
                    RSD_TI = inputdata_test["RSD_TI"].copy()
                    RSD_TI = (model[0] * RSD_TI) + model[1]
                    inputdata_test["adjTI_RSD_TI"] = RSD_TI
                items_adjected.append(inputdata_test)
        del temp
        className += 1

    adjusted_data = items_adjected[0]
    for item in items_adjected[1:]:
        adjusted_data = pd.concat([adjusted_data, item])
    results = post_adjustment_stats(inputdata_test, results, "Ref_TI", "adjTI_RSD_TI")

    results["adjustment"] = ["SS-SS"] * len(results)
    results = results.drop(columns=["sensor", "height"])

    return inputdata_test, results, m, c
