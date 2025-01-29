# api/routes/data.py
import opendatasets as od
from fastapi import APIRouter, HTTPException, Body
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import json
import sys
import os
import importlib

router = APIRouter()

@router.get("/download-iris-dataset")
def download_iris():
    dataset_url = "https://www.kaggle.com/datasets/uciml/iris"
    target_folder = "src/data"

    # Assurez-vous que le dossier existe
    os.makedirs(target_folder, exist_ok=True)

    try:
        # Téléchargez le dataset avec opendatasets
        print(f"Downloading the dataset from {dataset_url}...")
        od.download(dataset_url, data_dir=target_folder)

        return {"message": "Dataset downloaded successfully!", "path": target_folder}
    except Exception as e:
        # Gérer les erreurs en cas d'échec
        raise HTTPException(status_code=500, detail=f"Error downloading dataset: {str(e)}")
    
    router = APIRouter()

# Define the endpoint to load the dataset
@router.get("/load-iris-dataset", response_model=dict)
async def get_iris_data():
    try:
        # Path to the dataset file
        dataset_path = "src/data/iris/Iris.csv"

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Convert DataFrame to JSON format
        data = df.to_dict(orient="records")  # Each row is converted to a dictionary

        return {"data": data}  # Return the dataset as JSON
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preprocess-iris-dataset")
def preprocess_iris_dataset():
    dataset_path = os.path.join("src/data/iris/Iris.csv")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Please download it first.")
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Clean up column names (strip whitespace and convert to lowercase)
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop(columns=["id"], errors="ignore")

        # Check if the 'species' column exists after cleaning
        if 'species' not in df.columns:
            raise HTTPException(status_code=500, detail="The dataset does not contain a 'species' column.")

        # Preprocessing: Remove "Iris-" prefix from 'species' column
        df['species'] = df['species'].str.replace('Iris-', '', regex=False)

        # Return the processed dataset as JSON
        return {
            "message": "Dataset processed successfully!",
            "processed_data": df.to_dict(orient='records')  # Convert the DataFrame to a list of dictionaries
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")
    
    
@router.get("/split-iris-dataset")
def split_iris_dataset(test_size: float = 0.2, random_state: int = 42):
    """
    Splits the Iris dataset into training and testing sets.

    Parameters:
    - test_size: Proportion of the dataset to include in the test split (default: 0.2).
    - random_state: Random seed for reproducibility (default: 42).

    Returns:
    - JSON with training and testing datasets.
    """
    try:
        # First preprocess the dataset by calling the preprocessing function
        preprocess_response = preprocess_iris_dataset()
        processed_data = preprocess_response["processed_data"]

        # Convert the processed data back into a DataFrame
        df = pd.DataFrame(processed_data)

        # Split the dataset
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

        # Convert splits to JSON serializable format
        train_json = train.to_dict(orient='records')
        test_json = test.to_dict(orient='records')

        return {
            "message": "Dataset split successfully!",
            "train": train_json,
            "test": test_json
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting dataset: {str(e)}")
    
@router.post("/train-classification-model")
def train_model():
    """
    Trains a classification model on the Iris dataset and saves the model.

    Returns:
    - JSON with a success message and model path.
    """
    # Paths
    config_path = "services/epf-flower-data-science/src/config/model_parameters.json"
    models_folder = "services/epf-flower-data-science/src/models"
    os.makedirs(models_folder, exist_ok=True)

    try:
        # Load the processed dataset
        preprocess_response = preprocess_iris_dataset()
        processed_data = preprocess_response["processed_data"]

        # Convert the processed data back into a DataFrame
        df = pd.DataFrame(processed_data)

        # Split the dataset into features (X) and target (y)
        X = df.drop(columns=["species"])
        y = df["species"]

        # Load model parameters from JSON
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Initialize the model
        if config["model"] == "RandomForestClassifier":
            model = RandomForestClassifier(**config["parameters"])
        else:
            raise HTTPException(status_code=500, detail="Unsupported model specified in configuration.")

        # Train the model
        model.fit(X, y)

        # Save the model
        model_path = os.path.join(models_folder, "iris_model.joblib")
        dump(model, model_path)

        return {"message": "Model trained and saved successfully!", "model_path": model_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.post("/predict-iris-species")
def predict_species(sepal_length: float = Body(..., embed=True),
    sepal_width: float = Body(..., embed=True),
    petal_length: float = Body(..., embed=True),
    petal_width: float = Body(..., embed=True)):
    """
    Predicts the species of the Iris flower based on the input features.
    The features should be a list of four floats: [sepal_length, sepal_width, petal_length, petal_width].
    
    Returns:
    - JSON with the predicted species.
    """
    try:
        # Check if model exists
        models_folder = "services/epf-flower-data-science/src/models"
        model_path = os.path.join(models_folder, "iris_model.joblib")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Trained model not found.")

        # Load the trained model
        model = load(model_path)
        print('model loaded')

        features = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Ensure the features are in the correct shape for prediction (1, 4)
        if len(features[0]) != 4:
            raise HTTPException(status_code=400, detail="Invalid input. Please provide 4 features.")
        else:
            print('number of features is good')

        # Make a prediction using the model
        prediction = model.predict(features)

        print("i've done my prediction")
        print(prediction)

        predicted_species = prediction[0]

        return {"predicted_species": predicted_species}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
    

