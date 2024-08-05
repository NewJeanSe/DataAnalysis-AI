from flask import Flask, jsonify, render_template_string, send_file
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import threading
import time
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# 모델 로드
model = load_model('models/lstmModel.h5')

# 데이터 로드 및 스케일러 초기화
data = pd.read_csv('data/data최종.csv')
features = ['공급력', '현재수요', '최대예측수요', '공급예비력', '공급예비율', '운영예비력', '운영예비율', '습도', '기온']
target = ['공급력', '현재수요', '최대예측수요', '공급예비력', '공급예비율', '운영예비력', '운영예비율']

scalers = {}
for feature in features:
    scalers[feature] = MinMaxScaler()
    scalers[feature].fit(data[[feature]])

# 시퀀스 생성 함수
def create_sequences(data, sequence_length):
    xs = []
    for i in range(len(data) - sequence_length + 1):
        x = data[i:i + sequence_length]
        xs.append(x)
    return np.array(xs)

# 실시간 데이터 예측 함수
def get_current_data_and_predict():
    # 최신 데이터 로드 및 전처리
    data = pd.read_csv('data/data최종.csv')
    data['일시'] = pd.to_datetime(data['일시'])
    data = data.dropna()

    for feature in features:
        data[feature] = scalers[feature].transform(data[[feature]])

    sequence_length = 36
    X = create_sequences(data[features].values, sequence_length)

    predictions = []
    current_seq = X[-1]  # 최신 시퀀스를 사용하여 예측 시작

    for _ in range(36):  # 3시간 후까지 5분 단위 예측 (36개의 타임스텝)
        pred = model.predict(current_seq.reshape(1, sequence_length, len(features)))
        pred_unscaled = np.zeros_like(pred)
        for i, feature in enumerate(target):
            pred_unscaled[:, i] = scalers[feature].inverse_transform(pred[:, i].reshape(-1, 1)).reshape(-1)
        predictions.append(pred_unscaled.flatten().tolist())

        # 새로운 예측값을 시퀀스에 추가 (전체 features 중 예측된 target 값을 포함하도록 수정)
        new_seq = np.concatenate((current_seq[1:], np.zeros((1, len(features)))), axis=0)
        new_seq[-1, :len(target)] = pred.flatten()
        current_seq = new_seq

    return predictions

predictions = []

# 예측 업데이트 함수
def update_predictions():
    global predictions
    while True:
        predictions = get_current_data_and_predict()
        print(predictions)  # 예측 데이터 출력
        time.sleep(300)

# Flask 라우트 설정
@app.route('/predict')
def predict():
    return jsonify(predictions)

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-Time Predictions</title>
        <link rel="icon" href="/favicon.ico" type="image/x-icon">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/moment@2.29.1"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment@1.0.0"></script>
    </head>
    <body>
        <h1>Real-Time Predictions</h1>
        <div>
            <label><input type="checkbox" id="공급력" checked> 공급력</label>
            <label><input type="checkbox" id="현재수요" checked> 현재수요</label>
            <label><input type="checkbox" id="최대예측수요" checked> 최대예측수요</label>
            <label><input type="checkbox" id="공급예비력" checked> 공급예비력</label>
            <label><input type="checkbox" id="공급예비율" checked> 공급예비율</label>
            <label><input type="checkbox" id="운영예비력" checked> 운영예비력</label>
            <label><input type="checkbox" id="운영예비율" checked> 운영예비율</label>
        </div>
        <canvas id="predictionChart" width="800" height="400"></canvas>
        <script>
            var ctx = document.getElementById('predictionChart').getContext('2d');
            var predictionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [], // 시간 라벨
                    datasets: [
                        {
                            label: '공급력',
                            data: [], // 공급력 예측 데이터
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1,
                            fill: false
                        },
                        {
                            label: '현재수요',
                            data: [], // 현재수요 예측 데이터
                            borderColor: 'rgba(192, 75, 192, 1)',
                            borderWidth: 1,
                            fill: false
                        },
                        {
                            label: '최대예측수요',
                            data: [],
                            borderColor: 'rgba(192, 192, 75, 1)',
                            borderWidth: 1,
                            fill: false
                        },
                        {
                            label: '공급예비력',
                            data: [],
                            borderColor: 'rgba(75, 75, 192, 1)',
                            borderWidth: 1,
                            fill: false
                        },
                        {
                            label: '공급예비율',
                            data: [],
                            borderColor: 'rgba(75, 192, 75, 1)',
                            borderWidth: 1,
                            fill: false
                        },
                        {
                            label: '운영예비력',
                            data: [],
                            borderColor: 'rgba(192, 75, 75, 1)',
                            borderWidth: 1,
                            fill: false
                        },
                        {
                            label: '운영예비율',
                            data: [],
                            borderColor: 'rgba(75, 192, 192, 0.5)',
                            borderWidth: 1,
                            fill: false
                        }
                    ]
                },
                options: {
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'minute',
                                tooltipFormat: 'll HH:mm'
                            },
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    ]
                }
            });

            // 체크박스 상태 변경 시 그래프 업데이트
            document.querySelectorAll('input[type=checkbox]').forEach(function(checkbox) {
                checkbox.addEventListener('change', function() {
                    updateChart();
                });
            });

            // 주기적으로 데이터 업데이트
            function updateChart() {
                fetch('/predict')
                    .then(response => response.json())
                    .then(data => {
                        console.log(data);  // 데이터 확인
                        if (data.length === 0) return;  // 데이터가 없는 경우 처리

                        // 현시각 기준 3시간 후부터 5분 단위로 시간 생성
                        var now = new Date();
                        var futureTimes = [];
                        for (var i = 1; i <= 36; i++) {
                            futureTimes.push(new Date(now.getTime() + i * 5 * 60 * 1000));
                        }

                        predictionChart.data.labels = futureTimes;

                        // Update datasets for each target variable
                        var targetIds = ['공급력', '현재수요', '최대예측수요', '공급예비력', '공급예비율', '운영예비력', '운영예비율'];
                        targetIds.forEach((id, index) => {
                            if (document.getElementById(id).checked) {
                                predictionChart.data.datasets[index].data = data.map(prediction => prediction[index]);
                            } else {
                                predictionChart.data.datasets[index].data = Array(36).fill(null); // or handle the unchecked state appropriately
                            }
                        });

                        predictionChart.update();
                    })
                    .catch(error => console.error('Error fetching prediction data:', error));
            }

            setInterval(updateChart, 300000); // 5분마다 업데이트
            updateChart(); // 페이지 로드 시 첫 업데이트
        </script>
    </body>
    </html>
    """)

# 파비콘 제공
@app.route('/favicon.ico')
def favicon():
    return send_file('path/to/your/favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    threading.Thread(target=update_predictions).start()
    app.run()
