import torch
from torchvision import models, transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

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

# 데이터 변환 정의 (그레이스케일)
data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 그레이스케일로 변환
        CenterAndScale((224, 224), padding=20),  # 필기체 부분 찾아서 크기 조정
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 그레이스케일 이미지의 평균 및 표준편차로 정규화
    ])


# 클래스 이름 정의
class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# ResNet 모델 불러오기 및 수정 (ResNet-18)
model_ft = models.resnet18(weights=None)
num_ftrs = model_ft.fc.in_features
model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 첫 번째 레이어를 그레이스케일 입력으로 수정
model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))

# 학습된 모델의 가중치 불러오기
model_path = 'alphabet_resnet18_6.pth'
model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model_ft.eval()

# 이미지 예측 함수 정의
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert('L')  # 'L' 모드를 사용해 그레이스케일로 변환
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Batch size of 1

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top2_prob, top2_catid = torch.topk(probabilities, 2)

    top2_labels = [(class_names[catid], prob.item()) for prob, catid in zip(top2_prob[0], top2_catid[0])]
    return top2_labels

# 이미지 시각화 함수 정의
def imshow(image_path, title=None):
    image = Image.open(image_path).convert('L')  # 'L' 모드를 사용해 그레이스케일로 변환
    image = data_transforms(image)
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.5])
    std = np.array([0.5])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()

# 예시 이미지로 예측 실행
image_path = 'result_img/upper_d.png'  # 예측할 이미지 경로
top2_labels = predict_image(image_path, model_ft, class_names)
print(f'The predicted labels for the image are: {top2_labels}')

# 예측 이미지 시각화
title = '\n'.join([f'{label}: {prob:.2f}' for label, prob in top2_labels])
imshow(image_path, title=f'Predicted:\n{title}')
