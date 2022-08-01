<h1>WOE and IV Calculation Plugin in Dataiku DSS</h1>

Weight of evidence (WOE) and Information value (IV) are often used in credit scoring to perform variable transformation and selection. It provides insight for feature selection and feature engineering, common associated with inputs for machine learning model (very often a logistic regression model).It is widely used in credit scoring to measure the separation of good vs bad customers.

The advantages of WOE transformation are

Handles missing values
Handles outliers
The transformation is based on logarithmic value of distributions. This is aligned with the logistic regression output function
No need for dummy variables
By using proper binning technique, it can establish monotonic relationship (either increase or decrease) between the independent and dependent variable

Credits to Sundar Krishnan for the amazing post and headstart. you can check out his codes here.

https://sundarstyles89.medium.com/weight-of-evidence-and-information-value-using-python-6f05072e83eb

