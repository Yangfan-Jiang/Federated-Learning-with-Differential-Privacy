# data preprocess
import numpy as np
import pandas as pd


def bank_preprocess():
    data = pd.read_csv("bank-full.csv", sep=';')
    # job,
    job_map = {"blue-collar": 1, "management": 2, "technician": 3, "admin.": 4,
               "services": 5, "retired": 6, "self-employed": 7, "entrepreneur": 8,
               "unemployed": 9, "housemaid": 10, "student": 11, "unknown": 12}
    marital_map = {"married": 1, "single": 2, "divorced": 3}
    edu_map = {"secondary": 1, "tertiary": 2, "primary": 3, "unknown": 4}
    default_map = {"no": 1, "yes": 2}
    housing_map = {"no": 1, "yes": 2}
    loan_map = {"no": 1, "yes": 2}
    contact_map = {"cellular": 1, "unknown": 2, "telephone": 3}
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    poutcome_map = {"unknown": 1, "failure": 2, "other": 3, "success": 4}

    data["job"] = data["job"].map(job_map)
    data["marital"] = data["marital"].map(marital_map)
    data["education"] = data["education"].map(edu_map)
    data["default"] = data["default"].map(default_map)
    data["housing"] = data["housing"].map(housing_map)
    data["loan"] = data["loan"].map(loan_map)
    data["contact"] = data["contact"].map(contact_map)
    data["month"] = data["month"].map(month_map)
    data["poutcome"] = data["poutcome"].map(poutcome_map)
    data["y"] = data["y"].map({"no": 0, "yes": 1})

    for i in data.columns:
        data[i] = data[i] / abs(data[i]).max()
    return data


if __name__ == '__main__':
    df = bank_preprocess()
    df = np.array(df, dtype=np.float32)
    np.save("bank-full.npy", df)
