from torch import device as torchDevice
from torch import tensor as torchTensor
from torch import float32 as torchFloat32
from torch import no_grad as torchNo_grad
from torch import save as torchSave
from torch.nn import Sequential as torchSequential
from torch.nn import Conv1d as torchConv1d
from torch.nn import BatchNorm1d as torchBatchNorm1d
from torch.nn import ReLU as torchReLU
from torch.nn import Dropout as torchDropout
from torch.nn import Linear as torchLinear
from torch.nn import AdaptiveAvgPool1d as torchAdaptiveAvgPool1d
from torch.nn import Flatten as torchFlatten
from torch.nn import CrossEntropyLoss as torchCrossEntropyLoss
from torch.nn.utils import clip_grad_norm_ as torchClip_grad_norm_
from torch.cuda import is_available as torchIs_available
from torch.optim import Adam as torchAdam
from torch.optim.lr_scheduler import CosineAnnealingLR as torchCosineAnnealingLR
from torch.utils.data import DataLoader as torchDataLoader
from torch.utils.data import TensorDataset as torchTensorDataset
from numpy import load as numpyLoad
from numpy import transpose as numpyTranspose
from json import load as jsonLoad
import matplotlib.pyplot as plt
import logging



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

with open('config.json', 'r', encoding='utf-8') as file:
    config = jsonLoad(file)

with open(config['dataset']['info'], 'r', encoding='utf-8') as file:
    info = jsonLoad(file)



device = torchDevice('cuda' if torchIs_available() else 'cpu')
logger.info(f'Для обучения модели будет использоваться: {device}')



logger.info('Загрузка обучающего набора')

tx = numpyLoad(config['npy']['tx'])
ty = numpyLoad(config['npy']['ty'])
vx = numpyLoad(config['npy']['vx'])
vy = numpyLoad(config['npy']['vy'])

tx = tx.reshape(info['train_len'], info['max_video_len'], 42 * 3)
vx = vx.reshape(info['val_len'], info['max_video_len'], 42 * 3)

tx = numpyTranspose(tx, (0, 2, 1))
vx = numpyTranspose(vx, (0, 2, 1))

tx = torchTensor(tx, dtype=torchFloat32).to(device)
ty = torchTensor(ty, dtype=torchFloat32).to(device)
vx = torchTensor(vx, dtype=torchFloat32).to(device)
vy = torchTensor(vy, dtype=torchFloat32).to(device)

trainData = torchDataLoader(torchTensorDataset(tx, ty), batch_size=config['learn']['batch_size'], shuffle=True)
valData = torchDataLoader(torchTensorDataset(vx, vy), batch_size=config['learn']['batch_size'], shuffle=True)



logger.info('Создание модели')

model = torchSequential(
    torchConv1d(in_channels=126, out_channels=63, kernel_size=5, padding=2),
    torchBatchNorm1d(63),
    torchReLU(),
    torchDropout(0.5),

    torchConv1d(63, 31, kernel_size=3, padding=1),
    torchBatchNorm1d(31),
    torchReLU(),
    torchDropout(0.5),
    
    torchAdaptiveAvgPool1d(1),
    torchFlatten(),
    
    torchLinear(31, info['phrases_len'])
).to(device)



logger.info('Создание функции потерь, оптимизатора, планировщика')

criterion = torchCrossEntropyLoss(label_smoothing=config['learn']['label_smoothing'])
optimizer = torchAdam(model.parameters(), lr=config['learn']['learn_rate'], weight_decay=config['learn']['weight_decay'])
scheduler = torchCosineAnnealingLR(optimizer, T_max=config['learn']['t_max'], eta_min=config['learn']['eta_min'])



logger.info(f'Модель имеет {sum(p.numel() for p in model.parameters())} параметров')



maxValAccuracy = 0
maxValAccuracyEpoch = 0

history = {
    'trainAccuracy': [],
    'trainLoss': [],
    'valAccuracy': [],
    'valLoss': [],
    'diffLoss': [],
    'lr': []
}

print('Обучение')

for epoch in range(config['learn']['epochs']):
    model.train()
    trainLoss = 0.0
    trainCorrect = 0

    for inputs, targets in trainData:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        torchClip_grad_norm_(model.parameters(), config['learn']['grad_clip'])
        optimizer.step()

        trainLoss += loss.item()
        trainCorrect += (targets.argmax(dim=1) == outputs.argmax(dim=1)).sum().item()
    
    trainAccuracy = 100 * trainCorrect / info['train_len']
    trainLoss = trainLoss / info['train_len']
    scheduler.step()

    model.eval()
    valLoss = 0.0
    valCorrect = 0

    with torchNo_grad():
        for inputs, targets in valData:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            valLoss += loss.item()
            valCorrect += (targets.argmax(dim=1) == outputs.argmax(dim=1)).sum().item()
        
    valAccuracy = 100 * valCorrect / info['val_len']
    valLoss = valLoss / info['val_len']

    if maxValAccuracy <= valAccuracy:
        maxValAccuracy = valAccuracy
        maxValAccuracyEpoch = epoch
        torchSave(model, config['path']['models'] + '/' + f'model_{valAccuracy:.2f}_acc_{epoch}_ep.pth')
    
    history['trainAccuracy'].append(trainAccuracy)
    history['trainLoss'].append(trainLoss)
    history['valAccuracy'].append(valAccuracy)
    history['valLoss'].append(valLoss)
    history['diffLoss'].append(abs(trainLoss - valLoss))
    history['lr'].append(optimizer.param_groups[0]['lr'])
    
    print(
        f'Эпоха {epoch:4d} | '
        f'train acc {trainAccuracy:7.2f}% | '
        f'val acc {valAccuracy:7.2f}% | '
        f'max val acc {maxValAccuracy:7.2f}% (эпоха {maxValAccuracyEpoch:4d})'
    )



plt.plot(history['trainLoss'], label='train loss')
plt.plot(history['valLoss'], label='val loss')
plt.plot(history['diffLoss'], label='diff loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Проверка на переобучение')

plt.savefig('Проверка на переобучение.png', dpi=300)
plt.show()

plt.plot(history['trainAccuracy'], label='train acc')
plt.plot(history['valAccuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Отслеживание точности')

plt.savefig('Отслеживание точности.png', dpi=300)
plt.show()

plt.plot(history['lr'], label='lr')
plt.xlabel('Epoch')
plt.ylabel('Learnrate')
plt.legend()
plt.grid(True)
plt.title('Скорость обучения')

plt.savefig('Скорость обучения.png', dpi=300)
plt.show()

input('Нажмите enter для завершения')
