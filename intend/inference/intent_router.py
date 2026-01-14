import joblib
import numpy as np
model_path = '../../models/intend_x_train.pkl'

INTENT_THRESHOLD = {
    "greeting": 0.6,
    "faq": 0.7,
    "policy": 0.7,
    "knowledge": 0.7,
    "out_of_scope": 0.8
}
