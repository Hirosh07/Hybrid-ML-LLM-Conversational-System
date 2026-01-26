import joblib
import numpy as np
model_path = '../../models/intend_x_train.pkl'
model = joblib.load(model_path)

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

def route_intent(text):
    intent, confidence = predict_intent(text)
    if intent == 'greeting':
        return "hello! How can I Help You"
    elif intent == 'faq':
        return "Routing to FAQ handler."
    elif intent == 'policy':
        return "Routing to Policy handler."
    elif intent == 'knowledge':
        return "Routing to Knowledge Base handler."
    else:
        return call_llm(text)
        
def call_llm(text):
    return "fallback response from LLM"

if __name__ == "__main__":
    while True:
        q = input("You: ")
        print(route_intent(q))