import pickle
from hashlib import sha256

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def unified_hash(obj):
    return sha256(str(obj).encode()).hexdigest()