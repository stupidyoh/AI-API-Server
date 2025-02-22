from typing import Union
from fastapi import FastAPI

import model

and_model = model.AndModel()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

@app.get("/train")
def train():
    and_model.train()
    return {"Result": 'OK'}

@app.get("/predict/left/{left_id}/right/{right_id}")
def predict(left_id: int, right_id: int):
    result = and_model.predict([left_id, right_id])
    return {"result": result}

