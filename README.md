# AI-API-Server
HW 5


논리연산(NOT, AND, OR, NAND, NOR, XOR, XNOR)
- Single-Layer Perceptron(SLP) : NOT, AND, OR, NAND, NOR
- Multi-Layer Perceptron(MLP) : XOR, XNOR

## 실행방법
1. 터미널에서 아래 명령어 실행:
```
> fastapi dev main.py
```
2. 로컬 서버 http://127.0.0.1:8000 에 접속
3. 로컬 주소 뒤에 '/training'을 추가해 각 논리연산별 모델 학습(OK 출력시 학습 완료)
```
> {local_address}/training
```
4. 로컬 주소 뒤에 '/predict' + '/{logic_type}' 을 통해 실행할 논리연산자 선택
    * 그 뒤에 '?x={int}&y={int}' 파라미터를 주어 입력값 넣기
    * 이후 출력 결과 확인

```
> {local_address}/predict/NOT?x=1
> {local_address}/predict/AND?x=0&y=0
> {local_address}/predict/XOR?x=0&y=1
```