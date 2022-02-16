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
Start the Flask application
```commandline
python predict_api/app.py
```
Create a post request to test or train model.
```python
import requests

def test():
    url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
    body = {
        "identity_path": 'test_identity.csv.zip',
        "transaction_path": 'test_transaction.csv.zip',
    }
    response = requests.post(url, data=body)
    print(response.json())


def train():
    url = 'http://127.0.0.1:1080/train'  # localhost and the defined port + endpoint
    body = {
        "identity_path": 'train_identity.csv.zip',
        "transaction_path": 'train_transaction.csv.zip',
    }
    response = requests.post(url, data=body)
    print(response.json())


def main():
    train()
    test()


if __name__ == '__main__':
    main()

```