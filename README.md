# AI_Project_team5
Team5의 인공지능프로젝트 기말 프로젝트

## Hyperparameters of our model
저희의 프로젝트에는 총 4종류의 모델(ResNet, VGGNet, DenseNet, DarkNet)이 사용됩니다.
각 모델들을 생성하고 Hyperparameter를 전달하는 방법은 다음과 같습니다.

### ResNet
```Python
ResNet(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = num_classes)
```
ResNet은 3가지 hyperparameter를 이용합니다. block은 ResNet이 BasicBlock과 Bottleneck 중 어떤 종류의 블록을 이용할지 결정합니다. layers는 ResNet의 레이어들이 어떤 식으로 구성될지 정합니다. 우리의 프로젝트에서는 가장 기본적인 형태의 ResNet을 사용하기 위해 [2, 2, 2, 2]를 이용했습니다. num_classes는 데이터셋이 몇 개의 클래스로 이루어져 있는지를 전달해주어야 합니다.

### VGGNet
```Python
VGGNet(num_classes = num_classes)
```
VGGNet은 1개의 hyperparameter만 이용합니다. 데이터셋이 몇 개의 클래스로 이루어져 있는지 num_classes를 통해 전달해주면 됩니다.

### DenseNet
```Python
DenseNet(droprate = 0.2, block = BottleneckBlock, growth_rate = 12, num_classes=num_classes)
```
DenseNet(DenseNet-121)은 4개의 hyperparameter를 사용합니다. num_classes를 제외하고는 default로 설정되어 있기 때문에 실행할 때는 num_classes만 지정해도 충분합니다. num_classes는 데이터 내 class 종류를 말합니다. droprate은 dropout을 위해 사용합니다. block으로는 normal dense block, bottleneck block 중 원하는 것을 선택할 수 있습니다. 그러나 해당 프로젝트에서는 BottleneckBlock만을 구현하였습니다. 해당 모델에서는 dense block 반복 개수 = [6, 12, 24, 16] 즉 densenet-121을 사용하였습니다.growth_rate은 다음 블록으로 넘어갈 때 channel을 얼마나 증가시킬지 결정힙니다. 해당 프로젝트에서는 DenseNet 논문을 참고하여 12를 기본값으로 설정하였습니다.

### DarkNet
```Python
Darknet(num_classes = num_classes, block = ResidualBlock, droprate=0.2)
```
Darknet(Darknet-53)에서는 3rodml hyperparameter를 사용합니다. num_classes는 데이터셋 내 class의 개수를 뜻합니다. num_classes를 제외하고는 default로 설정되어 있기 때문에 실행할 때는 num_classes만 지정해도 충분합니다. block에서는 논문에서 언급된 대로 ResidualBlock을 사용합니다. 필요에 따라 직접 구현한 block을 넘겨줄 수 있으나 해당 프로젝트에서는 ResidualBlock이([1, 2, 8, 8, 4]개의 block을 순차적으로 사용) default입니다. Darknet에서는 batchnormalization을 사용하기 때문에 dropout을 사용하지 않아도 효과적으로 overfitting을 예방할 수 있다고 논문에 언급되어 있으나 다른 모델과의 비교 및 flexibility를 위해 droprate를 정할 수 있도록 하였습니다.

## Training
저희의 프로젝트에서 모델들을 학습시키려면 root/src/AI_train_model.py 파일을 실행시키면 됩니다.
해당 파일 내의 model 변수를 수정하는 것으로 어떤 종류의 모델을 학습시킬지 정할 수 있습니다.

## Testing
저희의 프로젝트에서 모델들을 테스트하려면 root/src/AI_test_model.py 파일을 실행시키면 됩니다. 해당 파일 내의 model 변수를 수정하는 것으로 어떤 종류의 모델을 테스트할지 정할 수 있습니다.

## Path to data
학습 및 테스트에 이용된 데이터들은 모두 루트디렉토리의 data 폴더에 위치해있습니다.

## Data source
저희의 데이터는 AI허브의 [해양 침적 쓰레기]([https://aihub.or.kr/aidata/30754](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=236)) 데이터셋을 다운로드하여 사용하였습니다.
