import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# # Load the model parameters into a model with the same architecture
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 10)
# model.load_state_dict(torch.load('./models/model_parameters.pth'))
# model.eval()  # Set the model to evaluation mode

# 데이터셋 불러오기
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 사전 훈련된 ResNet 모델 불러오기
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 10)  # CIFAR-10은 10개의 클래스

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 학습 루프
for epoch in range(2):  # 간단히 2 에포크만 학습
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i % 1000 == 999:
			print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}')
			running_loss = 0.0
print('Finished Training')

# Save only the model parameters
torch.save(net.state_dict(), './models/model_parameters.pth')