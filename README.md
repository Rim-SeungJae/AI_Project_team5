# AI_Project_team5
Team5의 인공지능프로젝트 기말 프로젝트

## Hyperparameters of our model
저희의 프로젝트에는 총 4종류의 모델(ResNet, VGGNet, DenseNet, DarkNet)이 사용됩니다.
각 모델들을 생성하고 Hyperparameter를 전달하는 방법은 다음과 같습니다.

### ResNet
'''Python
ResNet(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes)
'''
ResNet은 3가지 hyperparameter를 이용합니다. block은 ResNet이 BasicBlock과 Bottleneck 중 어떤 종류의 블록을 이용할지 결정합니다. layers는 ResNet의 레이어들이 어떤 식으로 구성될지 정합니다. 우리의 프로젝트에서는 가장 기본적인 형태의 ResNet을 사용하기 위해 [2, 2, 2, 2]를 이용했습니다. num_classes는 데이터셋이 몇 개의 클래스로 이루어져 있는지를 전달해주어야 합니다.

### VGGNet
'''Python
VGGNet(num_classes = num_classes)
'''
VGGNet은 1개의 hyperparameter만 이용합니다. 데이터셋이 몇 개의 클래스로 이루어져 있는지 num_classes를 통해 전달해주면 됩니다.

### DenseNet
'''Python
DenseNet(droprate = 0.2, block = BottleneckBlock, growth_rate = 12, num_classes=num_classes)
'''

### DarkNet
'''Python
Darknet(num_classes = num_classes, block = ResidualBlock, droprate)
'''
