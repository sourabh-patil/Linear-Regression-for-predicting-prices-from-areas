#########################################################################
#### Sourabh Patil 
#### 183079002 IIT Bombay
#### Kindly have a look at the code I have modified the given code.
#### I have made small changes. Thank you! Also I have commented the code
##########################################################################

import requests
import pandas
import scipy
import numpy as np
import sys
import pandas as pd

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


from scipy import stats 


# def predict_price(area):
#     """
#     This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

#     You can run this program from the command line using `python3 regression.py`.
#     """
#     response = requests.get(TRAIN_DATA_URL)
#     # YOUR IMPLEMENTATION HERE
#     print(response.status_code)


    # return 10



#### reading the train_data and getting dependent and independent variable seperately
response = requests.get(TRAIN_DATA_URL)
data = response.text
areas = data.split('price')[0].strip().split(',')[1::]
prices = data.split('price')[-1].strip().split(',')[1::]

#### Converting data to numpy array of type float
areas = np.array(areas).astype(np.float)
prices = np.array(prices).astype(np.float)

# print(np.max(areas))
# print(np.min(areas))
# print(np.max(prices))
# print(np.min(areas))

# print(areas)
# print(prices)


##### Using scipy for linear regression which will give us slope and intercept that we will get after training
slope, intercept, r_value, p_value, std_err = stats.linregress(areas,prices)

# print(slope)
# print(intercept)
# print(r_value)
# print(p_value)
# print(std_err)


###### Reading test data same as training data to verify the model
response_2 = requests.get(TEST_DATA_URL)
data_3 = response_2.text
areas_for_test = data_3.split('price')[0].strip().split(',')[1::]
areas_for_test = np.array(areas_for_test).astype(np.float)

actual_prices = data_3.split('price')[-1].strip().split(',')[1::]
actual_prices = np.array(actual_prices).astype(np.float)


######## Fitting the test data to the slope and intercept that we got while training 
predicted_prices = slope * areas_for_test + intercept


# print(len(predicted_prices))
# print(predicted_prices[:10])
# print(actual_prices[:10])

######### Calculation the rmse between predicted and actual prices 
rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))

print("The RMSE for the model trained is : {}".format(rmse))


# print("r sqrared {}".format(r_value**2))
# print(slope)
# print(intercept)

# response_2 = requests.get(TEST_DATA_URL)

# data = pd.read_csv('./regression/linreg_train.csv')

# print(data.head())


# YOUR IMPLEMENTATION HERE
# test = response_2.text
# areas_2 = test.split('price')[0].strip().split(',')[1::]
# prices_2 = test.split('price')[-1].strip().split(',')[1::]
# print(slope)
# print(intercept)
# if __name__ == "__main__":
#     # DO NOT CHANGE THE FOLLOWING CODE
#     from data import validation_data
#     areas = numpy.array(list(validation_data.keys()))
#     prices = numpy.array(list(validation_data.values()))
#     predicted_prices = predict_price(areas)
#     rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
#     try:
#         assert rmse < 170
#     except AssertionError:
#         # print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
#         sys.exit(1)
#     print("Success. RMSE = {}".format(rmse))
