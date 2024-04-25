from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

# Define request body schema using Pydantic BaseModel
class InputData(BaseModel):
    CHK_ACCT: str
    Duration: int
    History: str
    Purpose_of_credit: str
    Credit_Amount: int
    Balance_in_Savings_AC: str
    Employment: str
    Install_rate: int
    Marital_status: str
    # Co_applicant: str
    # Present_Resident: int
    Real_Estate: str
    Age: int
    Other_installment: str
    # Residence: str
    Num_Credits: int
    Job: str
    # No_dependents: int
    Phone: str
    Foreign: str

# Load the serialized model
with open('model_new.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def mapper(data):
    chk_acct = {'0DM': 0, 'less-200DM': 1, 'no-account': 2, 'over-200DM': 3}
    history = {'all-paid-duly':0, 'bank-paid-duly':1, 'critical':2, 'delay':3, 'duly-till-now':4}
    purpose_of_credit = {'business':0,'domestic-app':1,'education':2,'furniture':3,'new-car':4,'others':5,'radio-tv':6,'repairs':7,'retraining':8,'used-car':9}
    balance_in_savings_ac = {'less1000DM':0, 'less100DM':1, 'less500DM':2, 'over1000DM':3, 'unknown':4}
    employment = {'four-years':0, 'one-year':1, 'over-seven':2, 'seven-years':3, 'unemployed':4}
    marital_status = {'female-divorced': 0, 'male-divorced': 1, 'married-male':2, 'single-male':3}
    # co_applicant = {'co-applicant':0, 'guarantor':1, 'none':2}
    real_estate = {'building-society':0, 'car':1, 'none':2, 'real-estate':3}
    other_installment = {'bank':0, 'none':1, 'stores':2}
    # residence = {'free':0, 'own':1, 'rent':2}
    job = {'management':0, 'skilled':1, 'unemployed-non-resident':2, 'unskilled-resident':3}
    phone = {'no':0, 'yes':1}
    foreign = {'no':0, 'yes':1}
    return [[chk_acct[data.CHK_ACCT], 
            data.Duration, 
            history[data.History], 
            purpose_of_credit[data.Purpose_of_credit], 
            data.Credit_Amount, 
            balance_in_savings_ac[data.Balance_in_Savings_AC], 
            employment[data.Employment], 
            data.Install_rate, 
            marital_status[data.Marital_status],
            # co_applicant[data.Co_applicant],
            # data.Present_Resident,
            real_estate[data.Real_Estate],
            data.Age,
            other_installment[data.Other_installment],
            # residence[data.Residence],
            data.Num_Credits,
            job[data.Job],
            # data.No_dependents,
            phone[data.Phone],
            foreign[data.Foreign]
            ]]


@app.get("/")
def home():
    return FileResponse("templates/form.html")
    # return {"message": "Successfully hosted!"}

# Define API endpoint for making predictions
@app.post("/predict")
def predict(data: InputData):
    input_data = mapper(data)
    # columns = ['CHK_ACCT','Duration','History','Purpose of credit','Credit Amount','Balance in Savings A/C','Employment','Install_rate','Marital status','Co-applicant','Present Resident','Real Estate','Age','Other installment','Residence','Num_Credits','Job','No. dependents','Phone','Foreign']
    columns = ['CHK_ACCT','Duration','History','Purpose of credit','Credit Amount','Balance in Savings A/C','Employment','Install_rate','Marital status','Real Estate','Age','Other installment','Num_Credits','Job','Phone','Foreign']
    credit = {1:'good', 0:'bad'}
    new_df = pd.DataFrame(input_data, columns=columns)
    try:
        # Make prediction using the loaded model
        prediction = list(model.predict(new_df))
        return {"prediction": credit[prediction[0]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
