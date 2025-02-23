import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# SLP : NOT, AND, OR, NAND, NOR
class SLPModel:
    def __init__(self, logic):
        self.logic = logic
        self.inputs, self.outputs = self.select_logic()
        self.weights = np.random.rand(self.inputs.shape[1])
        self.bias = np.random.rand(1)
        self.learning_rate = 0.1
        self.epochs = 20

        self.load_parameters()

    def select_logic(self):
        if self.logic == "NOT":
            inputs = np.array([[0], [1]])
            outputs = np.array([1, 0])
            return inputs, outputs
        elif self.logic == "AND":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([0, 0, 0, 1])  
            return inputs, outputs
        elif self.logic == "OR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([0, 1, 1, 1])  
            return inputs, outputs
        elif self.logic == "NAND":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([1, 1, 1, 0])  
            return inputs, outputs
        elif self.logic == "NOR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([1, 0, 0, 0])  
            return inputs, outputs
        else:
            raise ValueError("Please choose your logical gate type in [NOT, AND, OR, NAND, NOR]")
    
    # 모델 파라미터 저장
    def save_parameters(self):
        params = {
            'weights': self.weights.tolist(),
            'bias': self.bias.tolist()
        }
        save_path = os.path.join(MODEL_DIR, f'slp_{self.logic.lower()}_params.json')
        with open(save_path, 'w') as f:
            json.dump(params, f)

    # 파라미터 불러오기
    def load_parameters(self):
        try:
            load_path = os.path.join(MODEL_DIR, f'slp_{self.logic.lower()}_params.json')
            with open(load_path, 'r') as f:
                params = json.load(f)
                self.weights = np.array(params['weights'])
                self.bias = np.array(params['bias'])
            return True
        except:
            return False

    # 모델 학습
    def training_model(self):       
        for epoch in range(self.epochs):
            for i in range(len(self.inputs)):
                # 총 입력 계산
                total_input = np.dot(self.inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = self.outputs[i] - prediction
                # print(f'inputs[i] : {self.inputs[i]}')
                # print(f'weights : {self.weights}')
                # print(f'bias before update: {self.bias}')
                # print(f'prediction: {prediction}')
                # print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += self.learning_rate * error * self.inputs[i]
                self.bias += self.learning_rate * error
                # print('====') 
        self.save_parameters()       

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    

# MLP : XOR, XNOR
class MLPModel(nn.Module):
    def __init__(self, logic):
        super(MLPModel, self).__init__()
        
        # 데이터 준비
        self.logic = logic
        self.inputs, self.outputs = self.select_logic()

        # 학습 파라미터
        self.epochs = 5000
        self.learning_rate = 0.01

        # 모델 컴파일
        self.model = nn.Sequential(
            nn.Linear(self.inputs.shape[1], 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1), 
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.load_model()

    # pth 확장자로 모델 저장    
    def save_model(self):
        save_path = os.path.join(MODEL_DIR, f'mlp_{self.logic.lower()}.pth')
        torch.save(self.model.state_dict(), save_path)

    # 저장된 모델 불러오기
    def load_model(self):
        try:
            load_path = os.path.join(MODEL_DIR, f'mlp_{self.logic.lower()}.pth')
            self.model.load_state_dict(torch.load(load_path))
            self.model.eval()
            return True
        except:
            return False

    def select_logic(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        if self.logic == "XOR":
            outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)  
            # return torch.tensor(inputs), torch.tensor(outputs)
        elif self.logic == "XNOR":
            outputs = np.array([[1], [0], [0], [1]], dtype=np.float32)  
        return inputs, outputs

    def training_model(self):
        # 학습을 위한 텐서화
        inputs = torch.tensor(self.inputs, dtype=torch.float32)
        outputs = torch.tensor(self.outputs, dtype=torch.float32)

        # epoch에 따른 학습 시행행
        for epoch in range(self.epochs):
            # self.model.train()
            self.optimizer.zero_grad()  # 옵티마이저의 변화도(gradient)를 초기화
            preds = self.model(inputs)  # 모델에 입력 데이터를 넣어 출력 계산
            loss = self.criterion(preds, outputs)  # 출력과 실제 레이블을 비교하여 손실 계산
            loss.backward()  # 역전파를 통해 손실에 대한 그래디언트 계산
            self.optimizer.step()  # 옵티마이저가 매개변수를 업데이트

            if (epoch + 1) % 100 == 0:  # 100 에포크마다 손실 출력
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        
        # 모델 파라미터 저장
        self.save_model()
    
    def predict(self, input_datas):
        self.model.eval()  # 모델을 평가 모드로 전환
        with torch.no_grad():  # 평가 중에는 그래디언트를 계산하지 않음
            input_tensor = torch.tensor(input_datas, dtype=torch.float32).reshape(1, -1)
            outputs = self.model(input_tensor)  # 모델에 입력 데이터를 전달하여 출력값 계산
            predicted = (outputs > 0.5).float()  # 출력값이 0.5보다 크면 1, 아니면 0으로 변환 (이진 분류)
            # print(outputs, predicted)
        return predicted.item()

        