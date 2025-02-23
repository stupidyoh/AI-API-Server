from typing import Union
from fastapi import FastAPI

import model

app = FastAPI()

# 모델 인스턴스 저장
models = {}

@app.get("/")
def read_root():
    return {"HW5": "Training and Test for Logical Operation[NOT, AND, OR, NAND, NOR, XOR, XNOR]"}

@app.get("/training")
def train_model():
    # Training SLP
    for logic in ["NOT", "AND", "OR", "NAND", "NOR"]:
        models[logic] = model.SLPModel(logic)
        models[logic].training_model()
        print(f"Training well for {logic}")

    # Training MLP
    for logic in ["XOR", "XNOR"]:
        models[logic] = model.MLPModel(logic)
        models[logic].training_model()
        print(f"Training well for {logic}")

    print("Trained models: ", models)
        
    return {"Result": 'OK'}

# /predict/NOT?x=1
# /predict/NOR?x=0&y=1
@app.get("/predict/{logic}")
def predict(logic: str, x: int, y: Union[int, None] = None):
    # 학습된 모델이 아닌 경우
    if logic not in models:
        return {"error": "Invalid logic gate"}

    # 학습된 모델에 대한 결과 출력
    if logic == "NOT":
        result = models[logic].predict([x])
    else:
        result = models[logic].predict([x, y])
    return {"result": result}

