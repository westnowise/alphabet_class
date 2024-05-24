import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

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

# 이미지가 있는 상위 폴더 경로
dataset_dir = '/Users/joseohyeon/Downloads/alphabet_class-master/alphabet_images'

# 하위 폴더 내의 모든 이미지 파일 목록 가져오기
image_paths = []
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_paths.append(os.path.join(root, file))

# 랜덤으로 5장 선택
selected_images = random.sample(image_paths, 5)

# 커스텀 전처리 적용 인스턴스 생성
center_and_scale = CenterAndScale((224, 224), padding=20)

# 선택된 이미지 열고 전처리
fig, axes = plt.subplots(5, 2, figsize=(8, 20))

for i, img_path in enumerate(selected_images):
    original_image = Image.open(img_path).convert('L')  # 그레이스케일로 변환
    processed_image = center_and_scale(original_image)
    
    # 원본 이미지 시각화
    axes[i, 0].imshow(original_image, cmap='gray')
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis('off')
    
    # 전처리된 이미지 시각화
    axes[i, 1].imshow(processed_image, cmap='gray')
    axes[i, 1].set_title("Processed Image")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
