# Import required libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load models
with open("model.pkl","rb") as file:
    model = pickle.load(file)

with open("scalar.pkl","rb") as file:
    scalar = pickle.load(file)

# FastAPI instance
app = FastAPI()

# Request data
class InsuranceInput(BaseModel):
    age: int
    sex: str # "male" or "female"	
    bmi: float	
    children: int
    smoker: str # "yes" or "no"
    region: str # "southeast" or "southwest" or "northwest" or "northeast"

## Defining the function to predict the output
@app.post("/predict")
def predict_charge(ins: InsuranceInput):
    try:
        # one-hot encoding
        sex = 0 if ins.sex == "male" else 1
        smoker = 0 if ins.smoker == "no" else 1
        region = 0 if ins.region == "northeast" else 1 if ins.region == "northwest" else 2 if ins.region == "southeast" else 3


        input_array = np.array([[ins.age,sex,ins.bmi,ins.children,smoker,region]])

        # Apply the same scaler used during training
        input_scaled = scalar.transform(input_array)

        prediction = model.predict(input_scaled)

        return {"predicted_charges": float(prediction[0][0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")