import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "intent_x_train.pkl")

model = joblib.load(MODEL_PATH)


INTENT_THRESHOLD = {
    "greeting": 0.5,
    "faq": 0.6,
    "policy": 0.6,
    "knowledge": 0.6,
    "out_of_scope": 0.7
}

def predict_intent(text):
    probs = model.predict_proba([text])[0]
    intent = model.classes_[np.argmax(probs)]
    max_prob = probs.max()

    print(f"DEBUG → intent={intent}, confidence={max_prob:.3f}")

    threshold = INTENT_THRESHOLD.get(intent, 0.5)

    if max_prob < threshold:
        return 'fallback_to_llm', max_prob

    return intent, max_prob

def route_intent(text: str):
    intent, confidence = predict_intent(text)

    if intent == "greeting":
        response = "Hello! How can I help you?"

    elif intent == "faq":
        response = "Routing to FAQ handler."

    elif intent == "policy":
        response = "Routing to Policy handler."

    elif intent == "knowledge":
        response = "Routing to Knowledge Base handler."

    else:
        response = "fallback response from LLM"

    result = {
        "intent": str(intent),
        "confidence": float(confidence),
        "response": response
    }

    print("DEBUG → API response:", result)
    return result
        
def call_llm(text):
    return "fallback response from LLM"

if __name__ == "__main__":
    while True:
        q = input("You: ")
        print(route_intent(q))