from torch import device as torchDevice
from torch.cuda import is_available as torchIs_available
from torch import load as torchLoad
from torch import tensor as torchTensor
from torch import float32 as torchFloat32
from torch import no_grad as torchNo_grad
from json import load as jsonLoad
from numpy import transpose, argmax
import logging

from processing import processingVideo as processingProcessingVideo



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



testFilesName = []



device = torchDevice('cuda' if torchIs_available() else 'cpu')
logger.info(f'Для обучения модели будет использоваться: {device}')



logger.info('Загрузка модели')

model = torchLoad(config['path']['models'] + '/' + config['main']['model'], weights_only=False).to(device)
model.eval()



logger.info('Чтение имён файлов тестового набора')

with open(config['dataset']['test'], 'r', encoding='utf-8') as file:
    for line in file:
        if line != '\n':
            currentLine = line.rstrip().split(' ')
            testFilesName.append(currentLine)



for file in testFilesName:
    x = processingProcessingVideo(name=config['video']['test'] + '/' + file[0])
    x = x.reshape(info['max_video_len'], x.shape[1] * x.shape[2])
    x = transpose(x, (1, 0))
    x = torchTensor(x, dtype=torchFloat32).unsqueeze(0).to(device)

    print('Истинное значение:', file[1])
    
    with torchNo_grad():
        outputs = model(x)
        prediction = outputs.cpu().numpy()[0]
        phrasesIndex = argmax(prediction)
        
        print('Предсказанное значение:', info['phrases'][phrasesIndex])
        print()

input('Нажмите enter для завершения')
