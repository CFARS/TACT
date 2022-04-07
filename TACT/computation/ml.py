import numpy as np


def machine_learning_TI(x_train, y_train, x_test, y_test, mode, TI_test):

    if len(x_train.shape) == 1:
        y_train = np.array(y_train).reshape(-1, 1).ravel()
        y_test = np.array(y_test).reshape(-1, 1).ravel()
        x_train = np.array(x_train).reshape(-1, 1)
        x_test = np.array(x_test).reshape(-1, 1)
        TI_test = np.array(TI_test).reshape(-1, 1)
    if len(x_train.shape) != 1 and x_train.shape[1] == 1:
        y_train = np.array(y_train).reshape(-1, 1).ravel()
        y_test = np.array(y_test).reshape(-1, 1).ravel()
        x_train = np.array(x_train).reshape(-1, 1)
        x_test = np.array(x_test).reshape(-1, 1)
        TI_test = np.array(TI_test).reshape(-1, 1)
    else:
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        TI_test = np.array(TI_test)

    if "RF" in mode:
        from sklearn.ensemble import RandomForestRegressor

        rfc_new = RandomForestRegressor(random_state=42, n_estimators=100)
        # rfc_new = RandomForestRegressor(random_state=42,max_features=2,n_estimators=100)
        rfc_new = rfc_new.fit(x_train, y_train.ravel())
        TI_pred = rfc_new.predict(x_test)

    if "SVR" in mode:
        from sklearn.svm import SVR

        clf = SVR(C=1.0, epsilon=0.2, kernel="poly", degree=2)
        clf.fit(x_train, y_train)
        TI_pred = clf.predict(x_test)

    if "MARS" in mode:
        from pyearth import Earth

        MARS_model = Earth()
        # MARS_model = Earth(max_terms=8,max_degree=2)
        MARS_model.fit(x_test, y_test)
        TI_pred[mask2] = MARS_model.predict(x_test)
        print(MARS_model.summary())

    if "NN" in mode:
        # import stuff
        NN_model = None

    return TI_pred
