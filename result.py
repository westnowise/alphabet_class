import torch
from torchvision import transforms
from PIL import Image
import json

# 데이터 변환 정의
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 클래스 이름 로드
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# ResNet 모델 불러오기
model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))

# 학습된 모델의 가중치 불러오기
model_path = 'alphabet_resnet18_2.pth'
model_ft.load_state_dict(torch.load(model_path))
model_ft.eval()

# 이미지 예측 함수 정의
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Batch size of 1

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    return label

# 예시 이미지로 예측 실행
image_path = 'result_img/a_4.png'  # 예측할 이미지 경로
label = predict_image(image_path, model_ft, class_names)
print(f'The predicted label for the image is: {label}')
