#!/bin/python

import pandas as pd
import numpy as np

# Search and download a particular dataset by name via UCI API

# Pandas opens "default of credit card clients.xls" excel file, deletes first
# row at index 0. This row has column names X1, X2, X3...X22, X23, Y nonessential
# for DAI experiment. Then saves the changed file as a csv file. 

filename = "default of credit card clients.xls"

# Set row 1 as the header. Ignore 0 row since its nonessential to DAI experiment
df = pd.read_excel(filename, header = 1)

# Stop Pandas from creating extra column, dont write row names since index=False
df.to_csv("UCI_Credit_Card.csv", index = False)