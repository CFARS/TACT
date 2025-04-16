import pandas as pd

def post_adjustment_stats(self, data, results, ref_col, TI_col):

        if isinstance(data, pd.DataFrame):
            fillEmpty = False
            if ref_col in data.columns and TI_col in data.columns:
                model_adjTI = self.get_regression(data[ref_col], data[TI_col])
                name1 = "TI_regression_" + TI_col + "_" + ref_col
                results.loc[name1, ["m"]] = model_adjTI[0]
                results.loc[name1, ["c"]] = model_adjTI[1]
                results.loc[name1, ["rsquared"]] = model_adjTI[2]
                results.loc[name1, ["difference"]] = model_adjTI[3]
                results.loc[name1, ["mse"]] = model_adjTI[4]
                results.loc[name1, ["rmse"]] = model_adjTI[5]
            else:
                fillEmpty = True
        else:
            fillEmpty = True
        if fillEmpty:
            name1 = "TI_regression_" + TI_col + "_" + ref_col
            results.loc[name1, ["m"]] = "NaN"
            results.loc[name1, ["c"]] = "NaN"
            results.loc[name1, ["rsquared"]] = "NaN"
            results.loc[name1, ["difference"]] = "NaN"
            results.loc[name1, ["mse"]] = "NaN"
            results.loc[name1, ["rmse"]] = "NaN"
        return results
