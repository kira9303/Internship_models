The data given had several missing values. I prefered to drop out the missing columns with missing categorical values and using the clean dataset without missing categorical values.

The data was converter into features and target_label (price, In our case).

The data was checked for any missing data type values, If present, Was converted into the form which can be used for prediction.

3)Categorical data was encoded into format which can be used for prediction ahead.

4)All the numerical features were then compiled and bought together as a feature_dataset which contained input arrays for our prediction model.

5)The best relation between the features and the target_price was plotted using chi-squared test. The details of gathered information from the test are commented inside the code.

6)The was then split accordingly into training and testing. Further, Regression model followed by a neural network was used as models to be trained on our prepared dataset.

7)The test accuracy obtained from both the models was around 54-57%.

8)Further accuracy can be increased with more feature engineering and adding more features to the dataset that are more relevant to our traget_price.
