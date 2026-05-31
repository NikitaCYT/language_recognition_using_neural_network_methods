from tqdm import tqdm
from numpy import save as numpySave
from json import load as jsonLoad
import logging

from processing import processingProcessingVideo



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

with open('config.json', 'r', encoding='utf-8') as file:
    config = jsonLoad(file)



trainFilesName = []
valFilesName = []



logger.info('Чтение имён файлов обучающего набора')

# Чтение имён файлов тренировочного набора
with open(config['dataset']['train'], 'r', encoding='utf-8') as file:
    for line in file:
        if line != '\n':
            currentLine = line.rstrip().split(' ')[0]
            trainFilesName.append(currentLine)

# Чтение имён файлов валидационного набора
with open(config['dataset']['val'], 'r', encoding='utf-8') as file:
    for line in file:
        if line != '\n':
            currentLine = line.rstrip().split(' ')[0]
            valFilesName.append(currentLine)



logger.info('Обработка обучающего набора')

# Обработка тренировочного набора
progress = tqdm(trainFilesName, desc='Обработка тренировочного набора')
for name in progress:
    allFrames = processingProcessingVideo(name=config['video']['train'] + '/' + name)
    numpySave(config['path']['train'] + '/' + f'{name}.npy', allFrames)

# Обработка валидационного набора
progress = tqdm(valFilesName, desc='Обработка валидационного набора')
for name in progress:
    allFrames = processingProcessingVideo(name=config['video']['val'] + '/' + name)
    numpySave(config['path']['val'] + '/' + f'{name}.npy', allFrames)

input('Нажмите enter для завершения')
