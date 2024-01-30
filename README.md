## Overview
Machine Learning application of classifying people for bad habits based on medical indicators with extensive use of MLOps practices.

## Installation
### Prerequisites
The project is structured according to microservice architecture so make sure you have Docker installed on your computer.

### Clone repository & install dependencies
```sh
git clone https://github.com/chuvalniy/mlops-practices.git
pip install -r requirements.txt
```
### Create & update credentials
Create an *.env* file you project root directory and copy variables from *.env-example* file. By default, *.env-example* has settings to run project locally, so no need to update credentials.

The next step is to create credentials for S3 storage. Go to the to your user's directory (i.e. C:\Users\MyUser) and create a folder called *.aws*. In this directory create a file called *credentials* and put this into file.
```sh
[default]
aws_access_key_id=minioadmin
aws_secret_access_key=minioadmin
aws_bucket_name=arts

[admin]
aws_access_key_id=minioadmin
aws_secret_access_key=minioadmin
```

Caution: aws_bucket_name should have the same content as **AWS_S3_BUCKET**

After all these steps you should have the following directory path C:\Users\MyUser\.aws\credentials.

These are default credentials in case if you're running this project locally and didn't make any changes in *.env* file.

### Run Docker
Navigate to project root directory and run docker containers.
```sh
docker-compose up -d --build
```
### Create S3 Bucket in Minio
To make mlflow be able to store model artifacts in S3 we need to make a bucket in S3 storage. 

Navigate to Minio console, by default the link is http://localhost:9001/. 

In the console you can see Buckets tab so open it. Click Create new bucket and call it *arts*. The name should be the same as your **AWS_S3_BUCKET** variable in the *.env* file.

### Attention (Windows)
This step is only necessary if you intend to use your experiments to deploy an ML service in the future. The project should work without it, but the solution below may solve some of your problems.

If you want to serve mlflow models locally on you machine, you have to set **MLFLOW_S3_ENDPOINT_URL** additionally in your PowerShell so mlflow can connect to Minio S3.
```sh
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
mlflow models serve 
```

## How to use
If you installed everything correctly, then this step will be simple.

### Execute pipeline
Execute this in project's root directory.
```sh
dvc pull
```

Run machine learning training pipeline.
```sh
dvc repro
```
### [Optional] Change model & tune hypeparameters.
You can choose your own hyperparameters or change the model (Random Forest by default) by modifying  **train.py** file. 
```sh
# Define parameters and model.
params = {
    "max_depth": 3,
    "n_estimators": 100,
    "random_state": RANDOM_STATE
}
model = RandomForestClassifier(**params)
```

## Documenation
In general, all the code is covered with docstrings and comments about what each component does, but there are some points that cannot be particularly described. Below is a description of the architecture, tech stack used and the data source.
### Architecture
If you want to check app architecture I suggest you to visit [this](docs/architecture.png) link.

### Stack
A more detailed description of each library that was used to create this application can be found [here](docs/stack.md).

### Data
Training data can be found on [Kaggle](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset). If you are interested in exploratory data analysis, you can find it at this [link](notebooks/) in two Jupyter Notebooks.
## Testing
Almost every function is provided with unit test via [pytest](https://docs.pytest.org/en/stable/contents.html) and [Click](https://github.com/pallets/click) libraries.

Execute the following command in your project directory to run the tests. 

```python
pytest -v
```

## License
// add

