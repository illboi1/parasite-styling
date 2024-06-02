import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

# Faster R-CNN 모델 불러오기
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 이미지 불러오기 및 전처리
url = "https://img.hankyung.com/photo/202001/AKR20200110125500005_01_i.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
transform = T.Compose([T.ToTensor()])
img = transform(img)

# 예측
with torch.no_grad():
	prediction = model([img])

# Print detected objects
for idx, box in enumerate(prediction[0]['boxes']):
	label = prediction[0]['labels'][idx].item()
	score = prediction[0]['scores'][idx].item()
	print(f"Object {idx+1}: Label={label}, Score={score}, Box={box}")

# 예측 결과 시각화
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img.permute(1, 2, 0))

for element in prediction[0]['boxes']:
	xmin, ymin, xmax, ymax = element
	rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
	ax.add_patch(rect)

plt.show()