# Application Technology Stack

## Infrastructure
- **Docker**: Microservice containerization
- **Jupyter-notebook**: Exploratory data analysis
- **PostgreSQL**: Storage for machine learning parameters and metrics
- **pgAdmin**: Visual interface for PostgreSQL
- **Minio**: Storage for S3 that uses AWS interface
- **Google Drive**: Storage for the training data
- **DVC**: Data versioning and workflow manager
- **Github Actions**: CI/CD
- **Nginx**: Proxy for Minio

## Python Libraries
- **Scikit-learn**: Training machine learning models and data preprocessing
- **Pytest**: Testing machine learning training pipeline
- **Pandas**: Data manipulation and analysis
- **Dotenv**: Handling .env variables inside Python code
- **Click**: Library to run code using CLI (compatibility with DVC workflow manager)
- **MLFlow**: Tracking server for machine learning models
