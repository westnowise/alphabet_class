import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from PIL import Image, ImageOps

# 필기체 부분을 찾아서 가운데로 이동시키고 크기 조정하는 커스텀 변환 정의
class CenterAndScale:
    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding
    
    def __call__(self, img):
        # 이미지 반전 및 경계 상자 얻기
        inverted_image = ImageOps.invert(img)
        bbox = inverted_image.getbbox()
        
        if bbox:
            # 필기체 부분을 잘라내기
            cropped_image = img.crop(bbox)
        else:
            # 필기체가 없는 경우 원본 이미지를 사용
            cropped_image = img

        # 원본 비율을 유지한 채 크기 조정
        cropped_image.thumbnail(self.size, Image.Resampling.LANCZOS)
        
        # 224x224로 패딩 추가
        new_image = Image.new('L', self.size, (255))  
        upper_left = ((self.size[0] - cropped_image.size[0]) // 2, (self.size[1] - cropped_image.size[1]) // 2)
        new_image.paste(cropped_image, upper_left)
        
        return new_image

# 데이터 변환 정의
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 그레이스케일로 변환
        CenterAndScale((224, 224), padding=20),  # 필기체 부분 찾아서 크기 조정
        transforms.RandomRotation(10),  # 랜덤 회전 추가
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 그레이스케일 이미지의 평균 및 표준편차로 정규화
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 그레이스케일로 변환
        CenterAndScale((224, 224), padding=20),  # 필기체 부분 찾아서 크기 조정
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 그레이스케일 이미지의 평균 및 표준편차로 정규화
    ]),
}

# 이미지가 있는 상위 폴더 경로
dataset_dir = '/Users/joseohyeon/Downloads/alphabet_class-master/alphabet_images'

# 전체 데이터셋 로드
dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms['train'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# 클래스 이름 가져오기
class_names = dataset.classes

# ResNet 모델 불러오기 및 수정
model_ft = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model_ft.fc.in_features

# 첫 번째 레이어를 수정하여 그레이스케일 이미지를 받을 수 있도록 설정
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

# 모델을 GPU로 이동
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_ft = model_ft.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 학습률 스케줄러 추가
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 학습 함수 정의
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['epoch'].append(epoch)
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

    return model, history

# 이미지와 라벨 시각화 함수 정의
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5])
    std = np.array([0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp.squeeze(), cmap='gray')  # 그레이스케일 이미지를 시각화
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# 일부 이미지 시각화
def visualize_some_images():
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

if __name__ == '__main__':
    # 필요한 경우
    torch.multiprocessing.freeze_support()

    # 모델 학습
    num_epochs = 10  # Epochs를 늘림
    model_ft, history = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=num_epochs)

    # 모델 저장
    model_path = 'alphabet_resnet18_7.pth'
    torch.save(model_ft.state_dict(), model_path)

    # 설정 값 및 결과 저장
    results = {
        'num_epochs': num_epochs,
        'optimizer': str(optimizer),
        'criterion': str(criterion),
        'history': history
    }

    results_path = 'training_results_7.json'
    with open(results_path, 'w') as f:
        json.dump(results, f)

    print(f'Model saved to {model_path}')
    print(f'Results saved to {results_path}')

    # 일부 이미지 시각화
    visualize_some_images()
