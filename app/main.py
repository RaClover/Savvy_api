from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import spacy
import logging

app = FastAPI()

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify which origins are allowed to access the server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

class SMS(BaseModel):
    message: str

class TransactionDetails(BaseModel):
    message: str
    is_transaction: bool
    category: str = None
    transaction_type: str = None
    amount: str = None
    currency: str = None
    card: str = None
    merchant: str = None
    balance: str = None

# Load models
try:
    classifier_model = joblib.load("app/models/classifier/classifier_model.pkl")
    tfidf_vectorizer = joblib.load("app/models/classifier/tfidf_vectorizer.pkl")
    categorizer_model = joblib.load("app/models/categorizer/categorizer_model.pkl")
    ner_model = spacy.load("app/models/ner_model")
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    raise RuntimeError(f"Model loading failed: {e}")

@app.post("/analyze-sms/", response_model=TransactionDetails)
def analyze_sms(sms: SMS):
    try:
        vectorized_message = tfidf_vectorizer.transform([sms.message])
        is_transaction = classifier_model.predict(vectorized_message)[0]
        response = TransactionDetails(message=sms.message, is_transaction=is_transaction)

        if is_transaction:
            # Directly pass the raw message to the categorizer model
            transaction_type, category = categorizer_model.predict([sms.message])[0]
            response.category = category
            response.transaction_type = transaction_type

            doc = ner_model(sms.message)
            for ent in doc.ents:
                if ent.label_ == "AMOUNT":
                    response.amount = ent.text
                elif ent.label_ == "CURRENCY":
                    response.currency = ent.text
                elif ent.label_ == "CARD":
                    response.card = ent.text
                elif ent.label_ == "MERCHANT":
                    response.merchant = ent.text
                elif ent.label_ == "BALANCE":
                    response.balance = ent.text

    except Exception as e:
        logging.error(f"Error processing SMS: {e}")
        raise HTTPException(status_code=500, detail="Error processing SMS")

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
