import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import spacy
from deep_translator import GoogleTranslator
import re

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

translator = GoogleTranslator(source='auto', target='en')

# Custom dictionary for consistent translations
custom_translation_dict = {
    r"\bPick up\b": "Purchase",  # Match whole word only
    r"\bpick up\b": "Purchase",  # Handle case variations
    r"\bticket\b": "card",  # Correctly translate "ticket" to "card"
    r"\bTicket\b": "card",  # Handle case variations
}


def apply_custom_translations(message: str, custom_dict: dict) -> str:
    for term, correct_translation in custom_dict.items():
        message = re.sub(term, correct_translation, message)
    return message


def separate_currency(message: str) -> str:
    # Separate the currency symbol from the amount
    currency_patterns = [
        r'(\d+)(р|rub|usd|eur|gbp|jpy|cad|aud|chf|cny|sek|nzd|mxn|sgd|hkd|nok|krw|try|inr|brl|zar)',
        # Common currency codes
        r'(\d+)(\$|€|£|¥|₽)'  # Common currency symbols
    ]
    for pattern in currency_patterns:
        message = re.sub(pattern, r'\1 \2', message)
    return message


def extract_card(message: str) -> str:
    # Extract the last four digits of the card information using a regex pattern
    match = re.search(r'\b(?:VISA|MASTERCARD|CARD|MIR)(\d{4})\b', message)
    if match:
        return match.group(1)
    return None


@app.post("/analyze-sms/", response_model=TransactionDetails)
def analyze_sms(sms: SMS):
    try:
        # Log the original message
        logging.info(f"Original message: {sms.message}")

        # Translate the message to English
        translated_message = translator.translate(sms.message)

        # Log the translated message
        logging.info(f"Translated message (raw): {translated_message}")

        # Apply custom translations to the translated message
        translated_message = apply_custom_translations(translated_message, custom_translation_dict)

        # Separate the currency symbol from the amount
        translated_message = separate_currency(translated_message)

        # Log the translated message after applying custom translations and separating currency
        logging.info(f"Translated message (corrected): {translated_message}")

        # Vectorize the translated message
        vectorized_message = tfidf_vectorizer.transform([translated_message])
        is_transaction = classifier_model.predict(vectorized_message)[0]
        response = TransactionDetails(message=sms.message, is_transaction=is_transaction)

        if is_transaction:
            # Directly pass the translated message to the categorizer model
            transaction_type, category = categorizer_model.predict([translated_message])[0]
            response.category = category
            response.transaction_type = transaction_type

            doc = ner_model(translated_message)
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

            # If card was not detected by NER, apply custom extraction
            if response.card is None:
                response.card = extract_card(translated_message)

    except Exception as e:
        logging.error(f"Error processing SMS: {e}")
        raise HTTPException(status_code=500, detail="Error processing SMS")

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
