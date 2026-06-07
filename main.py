import torch
import sys
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms # 이미지 전처리를 위한 torchvision.transforms 모듈을 가져옵니다. 이 모듈은 이미지 데이터를 텐서로 변환하거나 정규화하는 등의 다양한 변환을 제공하여 모델이 데이터를 더 효과적으로 학습할 수 있도록 돕습니다.
from model import AlexNet


def train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs=10):
    model.train() # 모델을 훈련 모드로 설정합니다. 이는 Dropout과 BatchNorm과 같은 레이어가 훈련 중에 다르게 동작하도록 합니다. 예를 들어, Dropout은 훈련 중에 뉴런을 무작위로 비활성화하지만, 평가 모드에서는 모든 뉴런이 활성화됩니다.
    for epoch in range(num_epochs):
        running_loss = 0.0 # 에포크 동안 누적되는 손실 값을 저장하는 변수입니다. 각 배치의 손실을 이 변수에 더하여 에포크 전체의 평균 손실을 계산할 수 있습니다.
        correct = 0 # 모델이 올바르게 예측한 샘플의 수를 저장하는 변수입니다. 각 배치에서 모델의 예측과 실제 레이블을 비교하여 이 변수를 업데이트합니다.
        total = 0 # 모델이 예측한 총 샘플 수를 저장하는 변수입니다. 각 배치에서 레이블의 크기를 이 변수에 더하여 에포크 전체의 정확도를 계산할 수 있습니다.
        
        for batch_idx, (images, labels) in enumerate(trainloader): # 훈련 데이터 로더에서 배치 단위로 데이터를 가져옵니다. batch_idx는 현재 배치의 인덱스이고, images와 labels는 각각 입력 이미지와 해당 레이블을 포함하는 텐서입니다.
            images = images.to(device) 
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images) # 모델에 입력 이미지 배치를 전달하여 예측을 생성합니다. outputs는 모델의 출력 텐서로, 각 클래스에 대한 로짓(logit) 값을 포함합니다. 이 값은 소프트맥스(softmax) 함수를 적용하여 확률로 변환할 수 있습니다.
            loss = criterion(outputs, labels) # 모델의 출력과 실제 레이블 간의 손실을 계산합니다. criterion은 앞서 정의한 손실 함수(CrossEntropyLoss)로, 모델이 얼마나 잘 예측하는지를 평가합니다. loss는 현재 배치의 손실 값을 나타내는 텐서입니다.
            
            # Backward pass
            optimizer.zero_grad() # 옵티마이저의 기울기를 초기화합니다. PyTorch에서는 기본적으로 기울기가 누적되므로, 각 배치마다 기울기를 초기화하여 이전 배치의 기울기가 현재 배치에 영향을 미치지 않도록 합니다. optimizer는 앞서 정의한 Adam 옵티마이저입니다.
            loss.backward() # 손실에 대한 모델의 가중치의 기울기를 계산합니다. 이 단계에서는 역전파(backpropagation)가 수행되어 모델의 가중치가 손실을 줄이는 방향으로 업데이트될 수 있도록 기울기가 계산됩니다.
            optimizer.step() # 옵티마이저가 모델의 가중치를 업데이트합니다. 이 단계에서는 계산된 기울기를 사용하여 모델의 가중치가 손실을 줄이는 방향으로 조정됩니다. optimizer는 앞서 정의한 Adam 옵티마이저입니다.
            # optimizer.zero_grad()


            # Statistics 학습 중 손실(loss)과 정확도(accuracy)를 계산하여 출력합니다.
            running_loss += loss.item() # loss.item()은 현재 배치의 손실 값을 반환합니다. 이를 running_loss에 누적하여 에포크 전체의 평균 손실을 계산할 수 있습니다.
            _, predicted = torch.max(outputs.data, 1) # predicted = indices 만 사용하겠다는 뜻입니다. (values 필요없음)
            total += labels.size(0) # labels.size(0)은 현재 배치의 샘플 수를 반환합니다. 이를 total에 누적하여 에포크 전체의 정확도를 계산할 수 있습니다.
            correct += (predicted == labels).sum().item() # predicted와 labels를 비교하여 모델이 올바르게 예측한 샘플의 수를 계산합니다. (predicted == labels)는 각 샘플에 대해 예측이 실제 레이블과 일치하는지 여부를 나타내는 불리언 텐서를 생성합니다. .sum()은 일치하는 샘플의 수를 계산하고, .item()은 이 값을 Python 숫자로 반환합니다. 이를 correct에 누적하여 에포크 전체의 정확도를 계산할 수 있습니다.
            
            if (batch_idx + 1) % 100 == 0: # 100 배치마다 현재 에포크, 배치 인덱스, 그리고 손실 값을 출력합니다. 이렇게 하면 훈련 과정에서 모델이 어떻게 학습되고 있는지 모니터링할 수 있습니다.
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {loss.item():.4f}") # 현재 에포크(epoch)와 배치 인덱스(batch_idx), 그리고 손실(loss) 값을 출력합니다. epoch+1과 batch_idx+1을 사용하여 1부터 시작하는 인덱스를 출력합니다. loss.item()은 현재 배치의 손실 값을 반환하며, .4f는 소수점 네 자리까지 출력하도록 형식을 지정합니다.
        
        epoch_loss = running_loss / len(trainloader) # 에포크 전체의 평균 손실을 계산합니다. running_loss는 에포크 동안 누적된 손실 값이며, len(trainloader)는 에포크 동안 처리된 배치 수입니다. 이를 통해 에포크 전체의 평균 손실을 얻을 수 있습니다.
        epoch_acc = 100 * correct / total # 에포크 전체의 정확도를 계산합니다. correct는 모델이 올바르게 예측한 샘플의 수이고, total은 모델이 예측한 총 샘플 수입니다. 이를 100으로 곱하여 백분율로 표현합니다.
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n") # 에포크 전체의 평균 손실과 정확도를 출력합니다. epoch+1을 사용하여 1부터 시작하는 인덱스를 출력합니다. epoch_loss는 에포크 전체의 평균 손실 값이며, .4f는 소수점 네 자리까지 출력하도록 형식을 지정합니다. epoch_acc는 에포크 전체의 정확도 백분율이며, .2f는 소수점 두 자리까지 출력하도록 형식을 지정합니다.
        
        scheduler.step() # 학습률 스케줄러의 상태를 업데이트합니다. 이 단계에서는 CosineAnnealingLR 스케줄러가 현재 에포크에 따라 학습률을 조정합니다. T_max=10은 학습률이 10 에포크마다 한 사이클을 완성하도록 설정하는 매개변수입니다. scheduler.step()을 호출하여 스케줄러가 다음 에포크에 사용할 학습률을 계산하도록 합니다.


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using device: {device}\n")

    print("Python:", sys.executable)
    print("Torch:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    model = AlexNet(num_classes=10).to(device) 
    print(f"AlexNet 모델이 생성되었습니다.\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4) # Adam 옵티마이저를 사용하여 모델의 가중치를 업데이트합니다. model.parameters()는 모델의 모든 가중치 매개변수를 반환하며, lr=0.001은 학습률을 설정합니다. weight_decay=5e-4는 L2 정규화(가중치 감쇠)를 적용하여 모델이 과적합되는 것을 방지하는 데 도움을 줍니다.
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환합니다. CIFAR-10 이미지의 픽셀 값은 0에서 255 사이이지만, ToTensor()는 이를 0에서 1 사이로 정규화하여 모델이 더 빠르게 학습할 수 있도록 돕습니다.
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CIFAR-10의 평균과 표준편차(Standard Deviation)로 정규화 -1 ~ 1 사이로 조정됩니다. 이는 모델이 더 빠르게 수렴하도록 돕습니다.
    # ])

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(), # RandAugment는 이미지 데이터 증강을 위한 기술로, 다양한 변환을 무작위로 적용하여 모델이 더 일반화된 특징을 학습할 수 있도록 돕습니다. 이를 통해 모델이 다양한 이미지 변형에 대해 강건해질 수 있습니다.
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])
    
    print("CIFAR-10 데이터 로딩 중...")
    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform) # CIFAR-10 데이터셋을 다운로드하고 로드합니다. root="./data"는 데이터를 저장할 디렉토리를 지정합니다. train=True는 훈련 데이터셋을 로드하겠다는 의미입니다. download=True는 데이터셋이 로컬에 없으면 다운로드하겠다는 뜻입니다. transform=transform은 앞서 정의한 변환을 적용하여 데이터를 전처리합니다.
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )
    print(f"훈련 데이터셋 크기: {len(trainset)}\n")
    
    print("훈련 시작...\n")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    train_model(model, trainloader, criterion, optimizer, scheduler, device, num_epochs=10)
    
    print("훈련 완료!")
    
    # 모델 저장
    torch.save(model.state_dict(), "./model/alexnet_cifar10.pth")
    print("모델이 저장되었습니다: ./model/alexnet_cifar10.pth")