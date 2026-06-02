import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000): # 기본 분류 클래스 수는 1000개
        super().__init__()
        
        # 1. 특징 추출부 (Features): 이미지에서 선, 면, 형태 등의 특징을 뽑아냅니다.
        self.features = nn.Sequential(
            # 첫 번째 Convolution: 큰 이미지(224x224)를 쪼개어 특징을 찾음
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 두 번째 Convolution
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 세 번째, 네 번째, 다섯 번째 Convolution (AlexNet의 핵심 깊이)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 2. 풀링부: 고차원 데이터를 1차원으로 펼치기 좋게 압축
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 3. 분류기 (Classifier): 추출된 특징을 바탕으로 최종 무엇인지 정답을 맞힙니다.
        self.classifier = nn.Sequential(
            nn.Dropout(), # 과적합 방지를 위해 무작위로 뉴런을 끔
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), # 최종 예측 클래스 개수로 출력
        )

    def forward(self, x):
        print("input:", x.shape) # 입력 이미지의 모양을 출력하여 확인

        # 데이터가 모델을 통과하는 흐름(순전파)을 정의합니다.
        x = self.features(x)
        print("After features:", x.shape) # 특징 추출 후의 모양을 출력하여 확인

        x = self.avgpool(x)
        print("After avgpool:", x.shape) # 풀링 후의 모양을 출력하여 확인

        x = torch.flatten(x, 1) # 2차원 이미지를 1차원 한 줄로 쭉 펼치기
        print("After flatten:", x.shape) # 펼친 후의 모양을 출력하여 확인

        x = self.classifier(x)
        print("Output:", x.shape) # 분류기 통과 후의 모양을 출력하여 확인
        
        return x

# === 모델 생성 및 GPU 연산 테스트 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device) # 예를 들어 10개의 사물을 맞히는 모델로 설정

print(f"AlexNet 모델이 다음 디바이스에 로드되었습니다: {device}")

# 가짜 이미지 데이터 생성 (배치사이즈 1, RGB 3채널, 224x224 크기)
fake_image = torch.randn(1, 3, 224, 224).to(device)

# 모델에 이미지 입력 후 예측값 받아오기
output = model(fake_image)

print("\n=== 가상 이미지 통과 테스트 성공! ===")
print("최종 출력 텐서 모양(Shape):", output.shape) # [1, 10] -> 1개 이미지에 대한 10개 클래스의 확률 점수
