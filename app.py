from fastapi import FastAPI
from pydantic import BaseModel
from intend.inference.intent_router import route_intent

app = FastAPI(title="Hybrid Intent Router API")


class QueryRequest(BaseModel):
    text: str


class QueryResponse(BaseModel):
    intent: str
    confidence: float
    response: str


@app.post("/route_intent", response_model=QueryResponse)
def predict_intent_api(request: QueryRequest):
    return route_intent(request.text)

@app.get("/")
def health():
    return {"status": "API is running"}
