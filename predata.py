import numpy as np
import pandas as pd

if __name__ == '__main__':
    name = "zsdata.xls"
    gsdata = pd.read_excel("data/" + name)
    print(gsdata)
    print(gsdata.info())
    gsdata.dropna(inplace=True)
    gsdata = gsdata.drop(gsdata[gsdata["e血液分析"] == 0].index)
    gsdata = gsdata.drop(gsdata[gsdata["j细胞血片"] == 0].index)
    gsdata = gsdata.drop(gsdata[gsdata["k细胞髓片"] == 0].index)
    gsdata.drop_duplicates(["a1病案号"],inplace=True)

    print(gsdata.info())

    gsdata.to_excel("data/pre_zsdata.xlsx",index=None)
