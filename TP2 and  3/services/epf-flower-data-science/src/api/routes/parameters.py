from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.firestore import FirestoreClient

# Create the router for parameters
router = APIRouter()

# Initialize Firestore client instance
firestore_client = FirestoreClient()

class Parameter(BaseModel):
    n_estimators: Optional[int] = None  # Optional, as it can be updated or added
    criterion: Optional[str] = None 

# Endpoint to retrieve parameters
@router.get("/parameters", response_model=dict)
async def get_parameters():
    try:
        # Retrieve parameters using FirestoreClient
        collection_name = "parameters"
        document_name = "parameters"
        parameters = firestore_client.get(collection_name, document_name)

        # Return the retrieved parameters
        return parameters

    except ValueError as e:
        # Handle case where document does not exist
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        # Handle other errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    

# Endpoint to update parameters in Firestore
@router.put("/parameters", response_model=dict)
async def update_parameters(parameter: Parameter):
    try:
        # Retrieve current parameters from Firestore
        collection_name = "parameters"
        document_name = "parameters"
        parameters = firestore_client.get(collection_name, document_name)

        # Update the parameters if provided in the request body
        if parameter.n_estimators is not None:
            parameters["n_estimators"] = parameter.n_estimators
        if parameter.criterion is not None:
            parameters["criterion"] = parameter.criterion

        # Save the updated parameters back to Firestore
        firestore_client.update(collection_name, document_name, parameters)

        return {"message": "Parameters updated successfully", "updated_parameters": parameters}

    except ValueError as e:
        # Handle case where document does not exist
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        # Handle other errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Endpoint to add parameters to Firestore
@router.post("/parameters", response_model=dict)
async def add_parameters(parameter: Parameter):
    try:
        # Retrieve current parameters from Firestore
        collection_name = "parameters"
        document_name = "parameters"
        parameters = firestore_client.get(collection_name, document_name)

        # Add or update the parameters as needed
        if parameter.n_estimators is not None:
            parameters["n_estimators"] = parameter.n_estimators
        if parameter.criterion is not None:
            parameters["criterion"] = parameter.criterion

        # Save the updated parameters back to Firestore
        firestore_client.update(collection_name, document_name, parameters)

        return {"message": "Parameters added successfully", "updated_parameters": parameters}

    except ValueError as e:
        # Handle case where document does not exist
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        # Handle other errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
