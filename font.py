import os
from PIL import Image, ImageDraw, ImageFont

# TTF 파일들이 있는 폴더 경로
ttf_dir = '../font'

# 이미지 출력 루트 디렉토리
output_root_dir = 'alphabet_images'
if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)

# 폰트 사이즈 설정
font_size = 150

# 이미지 크기 설정
image_size = (300, 300)  # 조금 더 크게 설정

# 알파벳 리스트
uppercase_alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
lowercase_alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)]

# 디렉토리 탐색 및 TTF 파일 찾기
for root, dirs, files in os.walk(ttf_dir):
    for file in files:
        if file.lower().endswith('.ttf'):
            ttf_path = os.path.join(root, file)
            font = ImageFont.truetype(ttf_path, font_size)

            # 대문자 처리
            for character in uppercase_alphabet:
                # 대문자 폴더 생성
                char_dir = os.path.join(output_root_dir, 'uppercase', character)
                if not os.path.exists(char_dir):
                    os.makedirs(char_dir)

                # 이미지 생성 (배경 흰색, 텍스트 검정색)
                image = Image.new('RGB', image_size, 'white')
                draw = ImageDraw.Draw(image)

                # 텍스트 경계 상자 얻기
                bbox = draw.textbbox((0, 0), character, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 텍스트 위치 (가운데 정렬)
                position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)

                # 텍스트 그리기
                draw.text(position, character, font=font, fill='black')

                # 이미지 저장
                image_name = f'{os.path.splitext(file)[0]}.png'
                image_path = os.path.join(char_dir, image_name)
                image.save(image_path)

                print(f'Saved: {image_path}')

            # 소문자 처리
            for character in lowercase_alphabet:
                # 소문자 폴더 생성
                char_dir = os.path.join(output_root_dir, 'lowercase', character)
                if not os.path.exists(char_dir):
                    os.makedirs(char_dir)

                # 이미지 생성 (배경 흰색, 텍스트 검정색)
                image = Image.new('RGB', image_size, 'white')
                draw = ImageDraw.Draw(image)

                # 텍스트 경계 상자 얻기
                bbox = draw.textbbox((0, 0), character, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 텍스트 위치 (가운데 정렬)
                position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)

                # 텍스트 그리기
                draw.text(position, character, font=font, fill='black')

                # 이미지 저장
                image_name = f'{os.path.splitext(file)[0]}.png'
                image_path = os.path.join(char_dir, image_name)
                image.save(image_path)

                print(f'Saved: {image_path}')
