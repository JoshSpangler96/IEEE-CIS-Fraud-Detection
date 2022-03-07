# IEEE-CIS-Fraud-Detection

## Description
Machine Learning Application for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle Competition

## Data
Categorical Features 
- Transaction
- ProductCD
- card1 - card6
- addr1, addr2
- P_emaildomain
- R_emaildomain
- M1 - M9

Categorical Features 
- Identity
- DeviceType
- DeviceInfo
- id_12 - id_38
- The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).

You can read more about the data from this post by the competition host.

Files\
train_{transaction, identity}.csv - the training set\
test_{transaction, identity}.csv - the test set (you must predict the isFraud value for these observations)\
sample_submission.csv - a sample submission file in the correct format\

***Get the data***
```commandline
kaggle competitions download -c ieee-fraud-detection
```

## Usage
Start the Flask application using Docker
\
***Docker***
```commandline
docker pull registry.hub.docker.com/jdspangler96/ieee_fraud:1.0.2
docker run -p 1080:1080 ieee_fraud:1.0.2
```
Submit Transaction data (identity and transaction [demo data in data/demo_*]) and get a percent change of it being a fraudulant transaction as a result (0.5 = 50%)
\
***AWS Instance***
```commandline
http://ec2-34-222-94-203.us-west-2.compute.amazonaws.com:1080/
```
Submit Transaction data (identity and transaction [demo data in data/demo_*]) and get a percent change of it being a fraudulant transaction as a result (0.5 = 50%)
