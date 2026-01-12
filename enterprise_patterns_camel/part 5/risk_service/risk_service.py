from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.post("/risk/enrich")
def enrich_risk(transaction: dict):
    amount = transaction.get("amount", 0)

    if amount > 10000:
        score = 80
        category = "HIGH"
    elif amount > 5000:
        score = 50
        category = "MEDIUM"
    else:
        score = 10
        category = "LOW"

    transaction["risk"]["score"] = score
    transaction["risk"]["category"] = category
    transaction["metadata"]["riskEvaluatedAt"] = datetime.utcnow().isoformat() + "Z"

    return transaction
