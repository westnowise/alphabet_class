import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import json

# 데이터 변환 정의
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터 로드
data_dir = './alphabet_images'
lowercase_dir = 'D:/font_img/alphabet_images/lowercase'
uppercase_dir = 'D:/font_img/alphabet_images/uppercase'

lowercase_dataset = datasets.ImageFolder(lowercase_dir, data_transforms['train'])
uppercase_dataset = datasets.ImageFolder(uppercase_dir, data_transforms['train'])

# 합쳐진 데이터셋을 훈련과 검증 데이터셋으로 분할
combined_dataset = ConcatDataset([lowercase_dataset, uppercase_dataset])
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# DataLoader 정의
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 데이터 로드 함수
def load_data(loader):
    features, labels = [], []
    for inputs, targets in loader:
        inputs = inputs.view(inputs.size(0), -1)  # 2D로 변환
        features.append(inputs.numpy())
        labels.append(targets.numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels

if __name__ == '__main__':
    # 훈련 데이터 로드
    X_train, y_train = load_data(train_loader)
    # 검증 데이터 로드
    X_val, y_val = load_data(val_loader)

    # 랜덤 포레스트 모델 학습
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 예측 및 평가
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)

    print("Training accuracy:", accuracy_score(y_train, y_pred_train))
    print("Validation accuracy:", accuracy_score(y_val, y_pred_val))
    print("\nClassification Report:\n", classification_report(y_val, y_pred_val))

    # 모델 저장
    model_path = 'random_forest_model.pkl'
    joblib.dump(clf, model_path)

    # 설정 값 및 결과 저장
    results = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'val_accuracy': accuracy_score(y_val, y_pred_val),
        'classification_report': classification_report(y_val, y_pred_val, output_dict=True)
    }

    results_path = 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f)

    print(f'Model saved to {model_path}')
    print(f'Results saved to {results_path}')
