from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import joblib
import pandas as pd
import argparse
import os
from io import StringIO
import json

# Define columns name  after transform
columns_name = ['credit_score',
 'age',
 'tenure',
 'balance',
 'products_number',
 'credit_card',
 'active_member',
 'estimated_salary',
 'country_France',
 'country_Germany',
 'country_Spain',
 'gender_Female',
 'gender_Male',
 'churn']

def train(X_train, y_train, args):
    """
    Trains a logistic regression model on the training data, and saves the model to args.model_dir directory.

    Args:
    X_train: pandas DataFrame, containing the training features.
    y_train: pandas Series, containing the target variable for training.
    args: Namespace, contains the command line arguments.

    Returns:
    model: trained logistic regression model.
    """
    print(f"--C: {args.c_reg}")
    print(f"--penalty: {args.penalty}")
    
    # Create a logistic regression model
    model = LogisticRegression(random_state=42, C=args.c_reg, penalty=args.penalty)
 
    # Train the model on the training data
    model.fit(X_train, y_train)

    # Save the trained model to args.model_dir directory using joblib
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print(f"Model saved at: {path}")

    # Return the trained model
    return model


def test(model, X_test, y_test):
    """Test the model on a given test set.

    Args:
        model: A fitted scikit-learn estimator object.
        X_test: A pandas DataFrame containing the test features.
        y_test: A pandas DataFrame containing the test labels.

    Returns:
        None.

    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and classification report
    cv = RepeatedStratifiedKFold(n_splits=9, n_repeats=3, random_state=42)
    acc = cross_val_score(model, X_test, y_test, cv=cv, scoring="accuracy").mean()    
    report =  classification_report(y_test, y_pred)

    # Calculate AUC using ROC curve and probability estimates
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    lr_auc = auc(fpr, tpr)

    # Print test results
    print("Accuracy: {:.3f}".format(acc))
    print("Classification Report:\n", report)
    print("AUC: {:.3f}".format(lr_auc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--c_reg", type=float, default=1.0)
    parser.add_argument("--penalty", type=str, default="l2")

    args = parser.parse_args()

    # Read transform datasets from S3
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.test, "test.csv"))

    print("Train columns names: ", train_df.columns)
    print("Test columns names: ", test_df.columns)

    # Split data into features and target for train data
    X_train = train_df.drop("churn",axis=1)
    y_train = train_df.select("churn",axis=1)

    # Split data into features and target for test data
    X_test = test_df.drop("churn",axis=1)
    y_test = test_df.select("churn",axis=1)

    model = train(X_train, y_train, args)
    test(model, X_test, y_test)


def model_fn(model_dir):
    """Deserialize fitted model from model_dir."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(input_data, content_type):
    """Read the raw input data as CSV or JSON.

    Args:
    input_data: string, representing the raw input data.
    content_type: string, representing the content type header of the input data.

    Returns:
    A pandas DataFrame object with the input data.

    Raises:
    RuntimeError: If the content type is not supported by the script.
    """

    # Check the content type of the input data
    if content_type == "text/csv":
        # If it is a CSV file, read it using the pandas library
        df = pd.read_csv(StringIO(input_data))
    elif content_type == "application/json":
        # If it is a JSON file, parse it using the json library and convert it to a Polars DataFrame
        data = json.loads(input_data)
        df = pd.DataFrame(data["instances"])
        # Create a DataFrame with the features as rows
        df = pd.DataFrame(
            df["features"].to_list(), orient="row"
        )  # orient = 'row' specifies that the input data should be interpreted as rows
    else:
        raise RuntimeError("{} not supported by script!".format(content_type))

    df.columns = columns_name

    # Drop the "churn" column (if it exists) because it is the target variable and we don't want to include it in     the input data and Convert the resulting polars DataFrame to a pandas DataFrame and return it
    df = df.drop("churn",axis=1)
    return df


def predict_fn(input_data, model):
    """Default implementation of predict_fn"""
    prediction = model.predict(input_data)
    return prediction
