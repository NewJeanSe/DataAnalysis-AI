# DataAnalysis-AI

# Flask 예측 서버

이 프로젝트는 Flask를 사용하여 예측 모델을 웹 서버로 호스팅하고, 프론트엔드에서 실시간 예측 데이터를 시각화하는 예제입니다.

## 설치 방법

1. **프로젝트 클론 및 이동**
    ```bash
    git clone <repository-url>
    cd project_folder
    ```

2. **가상 환경 생성 및 활성화**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows에서는 `venv\Scripts\activate`
    ```

3. **필요한 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

4. **Flask 서버 실행**
    ```bash
    python app.py
    ```

5. **브라우저에서 서버 접속**
    - 서버가 실행된 후, 브라우저에서 `http://127.0.0.1:5000`에 접속합니다.

## 파일 설명

- `app.py`: Flask 서버 코드
- `models/lstmModel.h5`: 사전 학습된 모델 파일
- `dataset/data.csv`: 예측에 사용할 데이터 파일
- `requirements.txt`: 필요한 라이브러리 목록
