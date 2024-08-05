import numpy as np
import pandas as pd

buildingInfo = pd.read_csv("데이콘 사이드프로젝트/sample_submission.csv")
train = pd.read_csv("데이콘 사이드프로젝트/train.csv")
test = pd.read_csv("데이콘 사이드프로젝트/test.csv")
print(buildingInfo.shape, train.shape, test.shape)