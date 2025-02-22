import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# SLP : NOT, AND, OR, NAND, NOR
class SLPModel:
    def __init__(self, logic):
        self.logic = logic
        self.inputs, self.outputs = self.select_logic()
        self.weights = np.random.rand(self.inputs.shape[1])
        self.bias = np.random.rand(1)
        self.learning_rate = 0.1
        self.epochs = 20

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

    def train(self):       
        for epoch in range(self.epochs):
            for i in range(len(self.inputs)):
                # 총 입력 계산
                total_input = np.dot(self.inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = self.outputs[i] - prediction
                print(f'inputs[i] : {self.inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += self.learning_rate * error * self.inputs[i]
                self.bias += self.learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    

# MLP : XOR, XNOR
class MLPModel(nn.Module):
    def __init__(self, logic):
        super(MLPModel, self).__init__()
        # 모델 구조 정의
        self.layer1 = nn.Linear(2, 2)  # 첫 번째 선형 레이어, 입력 크기 2, 출력 크기 2
        self.layer2 = nn.Linear(2, 1)  # 두 번째 선형 레이어, 입력 크기 2, 출력 크기 1
        self.relu = nn.ReLU()  # ReLU 활성화 함수
        self.sigmoid = nn.Sigmoid()  # Sigmoid 활성화 함수

        # 데이터 준비
        self.logic = logic
        self.inputs, self.outputs = self.select_logic()

        # 학습 파라미터
        self.epochs = 1000
        self.learning_rate = 0.1

        # 모델 컴파일
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = self.relu(self.layer1(x))  # 첫 번째 레이어와 ReLU 적용
        x = self.sigmoid(self.layer2(x))  # 두 번째 레이어와 Sigmoid 적용
        return x

    def select_logic(self):
        if self.logic == "XOR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
            outputs = np.array([0, 1, 1, 0], dtype=np.float32)  
            return torch.tensor(inputs), torch.tensor(outputs)
        elif self.logic == "XNOR":
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            outputs = np.array([1, 0, 0, 1])  
            return torch.tensor(inputs), torch.tensor(outputs)

    def train(self):
        for epoch in range(self.epochs):
            self.train()  # 모델을 훈련 모드로 설정
            self.optimizer.zero_grad()  # 옵티마이저의 변화도(gradient)를 초기화
            outputs = self(self.inputs)  # 모델에 입력 데이터를 넣어 출력 계산
            loss = self.criterion(outputs, self.outputs)  # 출력과 실제 레이블을 비교하여 손실 계산
            loss.backward()  # 역전파를 통해 손실에 대한 그래디언트 계산
            self.optimizer.step()  # 옵티마이저가 매개변수를 업데이트

            if (epoch + 1) % 100 == 0:  # 100 에포크마다 손실 출력
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, input_datas):
        self.eval()  # 모델을 평가 모드로 전환
        with torch.no_grad():  # 평가 중에는 그래디언트를 계산하지 않음
            outputs = self(input_datas)  # 모델에 입력 데이터를 전달하여 출력값 계산
            # predicted = (outputs > 0.5).float()  # 출력값이 0.5보다 크면 1, 아니면 0으로 변환 (이진 분류)
            # accuracy = (predicted == self.outputs).float().mean()  # 예측값과 실제값을 비교하여 정확도 계산
            # loss = self.criterion(outputs, self.outputs)  # 손실 함수(크로스 엔트로피 손실)를 사용하여 손실 계산
            # print(f'Loss: {loss.item()}, Accuracy: {accuracy.item()}')  # 손실과 정확도 출력
        return outputs
    