from cv2 import VideoCapture
from pathlib import Path
from tqdm import tqdm
from json import load as jsonLoad
from json import dump as jsonDump
import logging



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

with open('config.json', 'r', encoding='utf-8') as file:
    config = jsonLoad(file)



# Создание папок для будущего использования
Path(config['path']['data']).mkdir(exist_ok=True)
Path(config['path']['models']).mkdir(exist_ok=True)
Path(config['path']['train']).mkdir(exist_ok=True)
Path(config['path']['val']).mkdir(exist_ok=True)



trainFiles = []
valFiles = []

allVideoLen = {}
maxVideoLen = 0



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



logger.info('Проверка файлов на существование обучающего набора')

# Проверка файлов на существование тренировочного набора фраз
progress = tqdm(trainFiles, desc='Проверка файлов на существование тренировочного набора фраз')
for file in progress:
    filePath = Path(config['video']['train'] + '/' + f'{file[0]}.mp4')
    if not(filePath.exists()):
        print(f'Файла {file[0]} не существует')
        input()

# Проверка файлов на существование валидационного набора фраз
progress = tqdm(valFiles, desc='Проверка файлов на существование валидационного набора фраз')
for file in progress:
    filePath = Path(config['video']['val'] + '/' + f'{file[0]}.mp4')
    if not(filePath.exists()):
        print(f'Файла {file[0]} не существует')
        input()




logger.info('Создание ключей для обучающего набора')

# Создание ключей для тренировочного набора фраз
progress = tqdm(trainFiles, desc='Создание ключей для тренировочного набора фраз')
for file in progress:
    allVideoLen[file[1]] = []

# Создание ключей для валидационного набора фраз
progress = tqdm(valFiles, desc='Создание ключей для валидационного набора фраз')
for file in progress:
    allVideoLen[file[1]] = []



logger.info('Поиск самого длинного видео обучающего набора')

# Поиск самого длинного видео из тренировочного набора
progress = tqdm(trainFiles, desc='Поиск самого длинного видео из тренировочного набора')
for file in progress:
    currentLen = 0
    cap = VideoCapture(config['video']['train'] + '/' + f'{file[0]}.mp4')

    while True:
        ret, _ = cap.read()
        if not ret:
            break
        currentLen += 1

    maxVideoLen = max(maxVideoLen, currentLen) # Обновляем максимальную длинну
    allVideoLen[file[1]].append(currentLen) # Записываем длинну видео

# Поиск самого длинного видео из валидационного набора
progress = tqdm(valFiles, desc='Поиск самого длинного видео из валидационного набора')
for file in progress:
    currentLen = 0
    cap = VideoCapture(config['video']['val'] + '/' + f'{file[0]}.mp4')

    while True:
        ret, _ = cap.read()
        if not ret:
            break
        currentLen += 1
    
    maxVideoLen = max(maxVideoLen, currentLen) # Обновляем максимальную длинну
    allVideoLen[file[1]].append(currentLen) # Записываем длинну видео



logger.info('Сохранение данных')

# Сохранение данных для будущего использования
jsonData = {}
phrases = []
avgVideoLen = {}

# Создание словаря фраз
phrases = [key for key in allVideoLen.keys()]

# Поиск средней длинны всех видео
for key in allVideoLen.keys():
    avgVideoLen[key] = sum(allVideoLen[key]) // len(allVideoLen[key])

jsonData['phrases'] = phrases
jsonData['phrases_len'] = len(phrases)
jsonData['max_video_len'] = maxVideoLen
jsonData['avg_video_len'] = avgVideoLen
jsonData['train_len'] = len(trainFiles)
jsonData['val_len'] = len(valFiles)

with open(config['dataset']['info'], 'w', encoding='utf-8') as file:
    jsonDump(jsonData, file, ensure_ascii=False, indent=4)

input('Нажмите enter для завершения')
