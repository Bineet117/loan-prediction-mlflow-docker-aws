

import pickle
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import traceback
from typing import Optional

# Define the input data structure using Pydantic
class LoanRequest(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    total_asset_value: float

# Initialize FastAPI app
app = FastAPI()

# Set the path to the working directory and models folder
working_dir = os.getcwd()
models_folder = os.path.join(working_dir, "models")
model_path = os.path.join(models_folder, "model.pkl")

# Load the trained model
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        raise e
else:
    print(f"Error: Model file not found at {model_path}")
    model = None  # Safeguard against unintentional usage

# Define the single prediction endpoint
@app.post("/predict")
def predict_loan(request: LoanRequest):
    try:
        # Check if the model is loaded
        if model is None:
            raise HTTPException(
                status_code=500, detail="Model is not loaded. Ensure the model file exists and is valid."
            )

        # Convert the incoming request data into a DataFrame
        data = request.dict()
        input_data = pd.DataFrame([data])

        # Perform prediction using the loaded model
        prediction = model.predict(input_data)

        # Convert the prediction to a JSON-serializable format
        prediction = prediction.tolist() if hasattr(prediction, "tolist") else prediction

        # Return the prediction result
        return {"prediction": prediction}

    except Exception as e:
        # Log the error traceback
        error_msg = traceback.format_exc()
        print(f"Error during prediction: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

# Define the batch prediction endpoint (CSV file upload)
@app.post("/batch_predict")
def batch_predict_loans(file: UploadFile = File(...)):
    try:
        # Check if the model is loaded
        if model is None:
            raise HTTPException(
                status_code=500, detail="Model is not loaded. Ensure the model file exists and is valid."
            )

        # Read the uploaded CSV file into a DataFrame
        try:
            input_data = pd.read_csv(file.file)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error reading the uploaded CSV file: {str(e)}"
            )

        # Perform predictions using the loaded model
        predictions = model.predict(input_data)

        # Convert the predictions to a JSON-serializable format
        predictions = predictions.tolist() if hasattr(predictions, "tolist") else predictions

        # Add predictions to the DataFrame
        input_data["prediction"] = predictions

        # Save the predictions to a CSV file (optional)
        output_file_path = os.path.join(working_dir, "predictions.csv")
        # input_data.to_csv(output_file_path, index=False)

        # Return the prediction results and file path
        return {
            "message": "Batch predictions completed successfully.",
            "predictions": predictions,
            "output_file_path": output_file_path
        }

    except Exception as e:
        # Log the error traceback
        error_msg = traceback.format_exc()
        print(f"Error during batch prediction: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during batch prediction: {str(e)}"
        )
