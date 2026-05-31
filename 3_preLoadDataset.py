from tqdm import tqdm
from numpy import load as numpyLoad
from numpy import save as numpySave
from numpy import zeros as numpyZeros
from json import load as jsonLoad
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



trainFiles = []
valFiles = []



logger.info('Чтение имён файлов и фраз обучающего набора')

# Чтение имён файлов и фраз тренировочного набора
with open(config['dataset']['train'], 'r', encoding='utf-8') as file:
    for line in file:
        if line != '\n':
            currentLine = line.rstrip().split(' ')
            trainFiles.append(currentLine)

# Чтение имён файлов и фраз валидационного набора
with open(config['dataset']['val'], 'r', encoding='utf-8') as file:
    for line in file:
        if line != '\n':
            currentLine = line.rstrip().split(' ')
            valFiles.append(currentLine)



logger.info('Преобразование обучающего набора')

tx = numpyZeros((info['train_len'], info['max_video_len'], 42, 3))
vx = numpyZeros((info['val_len'], info['max_video_len'], 42, 3))

# Преобразование тренировочного набора
progress = tqdm(trainFiles, desc='Преобразование тренировочного набора'); i = 0
for name in progress:
    tx[i] = numpyLoad(config['path']['train'] + '/' + name[0] + '.npy')
    i += 1

# Преобразование валидационного набора
progress = tqdm(valFiles, desc='Преобразование валидационного набора'); i = 0
for name in progress:
    vx[i] = numpyLoad(config['path']['val'] + '/' + name[0] + '.npy')
    i += 1



logger.info('Преобразование фраз в массив вероятностей обучающего набора')

ty = numpyZeros((info['train_len'], info['phrases_len']))
vy = numpyZeros((info['val_len'], info['phrases_len']))

# Преобразование фраз в массив вероятностей тренировочного набора
progress = tqdm(trainFiles, desc='Преобразование фраз тренировочного набора'); i = 0
for phrase in progress:
    tmpTy = numpyZeros((info['phrases_len']))
    tmpTy[info['phrases'].index(phrase[1])] = 1
    ty[i] = tmpTy
    i += 1

# Преобразование фраз в массив вероятностей валидационного набора
progress = tqdm(valFiles, desc='Преобразование фраз валидационного набора'); i = 0
for phrase in progress:
    tmpVy = numpyZeros((info['phrases_len']))
    tmpVy[info['phrases'].index(phrase[1])] = 1
    vy[i] = tmpVy
    i += 1



logger.info('Сохранение полного обучающего набора')

numpySave(config['npy']['tx'], tx) # Сохранение полного tx train набора
numpySave(config['npy']['ty'], ty) # Сохранение полного ty train набора
numpySave(config['npy']['vx'], vx) # Сохранение полного vx val набора
numpySave(config['npy']['vy'], vy) # Сохранение полного vy val набора

input('Нажмите enter для завершения')
