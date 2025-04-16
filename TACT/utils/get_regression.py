import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def get_regression(self, x, y):
        """
        Compute linear regression of data -> need to deprecate this function for get_modelRegression..
        """
        df = pd.DataFrame()
        df["x"] = x
        df["y"] = y
        df = df.dropna()

        feature_name = "x"
        target_name = "y"

        data, target = df[[feature_name]], df[target_name]

        if len(df) > 1:

            x = df["x"].astype(float)
            y = df["y"].astype(float)

            lm = LinearRegression()
            lm.fit(data, target)
            predict = lm.predict(data)

            result = [lm.coef_[0], lm.intercept_]  # slope and intercept?
            result.append(lm.score(data, target))  # r score?
            result.append(abs((x - y).mean()))  # mean diff?

            mse = mean_squared_error(target, predict, multioutput="raw_values")
            rmse = np.sqrt(mse)
            result.append(mse[0])
            result.append(rmse[0])

        else:
            result = [None, None, None, None, None, None]
            result = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        return result