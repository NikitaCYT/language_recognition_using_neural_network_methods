from torch import device as torchDevice
from torch.cuda import is_available as torchIs_available
from torch import load as torchLoad
from torch import tensor as torchTensor
from torch import float32 as torchFloat32
from torch import no_grad as torchNo_grad
from numpy import transpose as numpyTranspose
from numpy import argmax as numpyArgmax
from numpy import array as numpyArray
from tkinter import Tk as tkinterTk
from tkinter import Frame as tkinterFrame
from tkinter import Button as tkinterButton
from tkinter import Label as tkinterLabel
from tkinter import Entry as tkinterEntry
from PIL.Image import fromarray as pilFromarray
from PIL.ImageTk import PhotoImage as PILTkPhotoImage
from cv2 import VideoCapture
from cv2 import cvtColor as cv2CvtColor
from cv2 import COLOR_BGR2RGB as cv2COLOR_BGR2RGB
from cv2 import flip as cv2Flip
from time import time as timeTime
from json import load as jsonLoad
from rapidfuzz import fuzz
import logging

from processing import processingFrame as processingProcessingFrame
from processing import isHandsInFrame as processingIsHandsInFrame
from processing import normBuferFrames as processingNormBuferFrames


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



logger.info('Загрузка модели')

model = torchLoad(config['path']['models'] + '/' + config['main']['model'], weights_only=False).to(device)
model.eval()


cap = VideoCapture(0)
lastTime = timeTime()
runningCamera = 'no'

buferFrames = []
buferFrameRecognized = []
maxSkippingFrame = info['max_video_len'] // 2
currentSkippingFrame = 0
testNum = 0



def showFrame(frame):
    global runningCamera, buferFrame
    
    if str(frame) == '.!frame4': runningCamera = 'test'
    elif str(frame) == '.!frame5': runningCamera = 'main'
    else: runningCamera = 'no'

    buferFrame = []
    frame.tkraise()

def mainLoop():
    global lastTime

    if runningCamera == 'main': cameraMain()
    elif runningCamera == 'test': cameraTest()

    if runningCamera != 'no':
        delay = max(1, int(33 - (timeTime() - lastTime) * 1000))
        lastTime = timeTime()
        root.after(delay, mainLoop)
    else:
        root.after(500, mainLoop)

def cameraTest():
    global lastTime, buferFrames, currentSkippingFrame, buferFrameRecognized, testNum

    ret, frame = cap.read()
    if config['main']['mirror_frame']: frame = cv2Flip(frame, 1)
    if ret:
        frame = cv2CvtColor(frame, cv2COLOR_BGR2RGB)

        if processingIsHandsInFrame(frame=frame):
            frame, h = processingProcessingFrame(frame=frame, returnLMFrame=True)
            buferFrames.append(h)
            currentSkippingFrame = 0
        else:
            currentSkippingFrame += 1
            
        if currentSkippingFrame == maxSkippingFrame:
            currentSkippingFrame = 0

            while len(buferFrames) != 0:
                currentBuferFrames = buferFrames[:info['max_video_len']]
                    
                x = processingNormBuferFrames(numpyArray(currentBuferFrames))
                x = x.reshape(info['max_video_len'], x.shape[1] * x.shape[2])
                x = numpyTranspose(x, (1, 0))
                x = torchTensor(x, dtype=torchFloat32).unsqueeze(0).to(device)

                with torchNo_grad():
                    outputs = model(x)
                    prediction = outputs.cpu().numpy()[0]
                    phrasesIndex = numpyArgmax(prediction)
                    
                buferFrameRecognized.append(info['phrases'][phrasesIndex])

                buferFrames = buferFrames[info['avg_video_len'][info['phrases'][phrasesIndex]]:]

            if (buferFrameRecognized):
                screenTest_body_result.config(text = f'Тест {testNum}\n' + '\n'.join(buferFrameRecognized))
                testNum += 1
                
            buferFrames = []
            buferFrameRecognized = []

        photo = PILTkPhotoImage(image = pilFromarray(frame))
        screenTest_body_camera.configure(image = photo)
        screenTest_body_camera.image = photo

def cameraMain():
    global lastTime, buferFrames, currentSkippingFrame, buferFrameRecognized

    ret, frame = cap.read()
    if config['main']['mirror_frame']: frame = cv2Flip(frame, 1)
    if ret:
        frame = cv2CvtColor(frame, cv2COLOR_BGR2RGB)

        if processingIsHandsInFrame(frame=frame):
            frame, h = processingProcessingFrame(frame=frame, returnLMFrame=True)
            buferFrames.append(h)
            currentSkippingFrame = 0
        else:
            currentSkippingFrame += 1
            
        if currentSkippingFrame == maxSkippingFrame:
            currentSkippingFrame = 0

            while len(buferFrames) != 0:
                currentBuferFrames = buferFrames[:info['max_video_len']]
                    
                x = processingNormBuferFrames(numpyArray(currentBuferFrames))
                x = x.reshape(info['max_video_len'], x.shape[1] * x.shape[2])
                x = numpyTranspose(x, (1, 0))
                x = torchTensor(x, dtype=torchFloat32).unsqueeze(0).to(device)

                with torchNo_grad():
                    outputs = model(x)
                    prediction = outputs.cpu().numpy()[0]
                    phrasesIndex = numpyArgmax(prediction)
                    
                buferFrameRecognized.append(info['phrases'][phrasesIndex])

                buferFrames = buferFrames[info['avg_video_len'][info['phrases'][phrasesIndex]]:]

            if (buferFrameRecognized):
                outString = []
                for word in buferFrameRecognized:
                    if word != 'no_event':
                        outString.append(word)
                outString = removeDuplicates(outString)
                
                if outString: resultProcessing(outString=' '.join(outString))

                
            buferFrames = []
            buferFrameRecognized = []

def removeDuplicates(words: list):
    if not words: return []
    result = [words[0]]
    
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)
    
    return result

def resultProcessing(outString: str):
    logger.info(f'Распознано: {outString}')
    
    if fuzz.ratio(outString, 'где aудитория') >= config['main']['accuracy_comparison'] : showFrame(screenEnteringAudienceNumber)
    elif fuzz.ratio(outString, 'где библиотека') >= config['main']['accuracy_comparison'] : showFrame(screenLibraryLocation)
    elif fuzz.ratio(outString, 'где бухгалтерия') >= config['main']['accuracy_comparison'] : showFrame(screenAccountingLocation)
    elif fuzz.ratio(outString, 'где столовая') >= config['main']['accuracy_comparison'] : showFrame(screenDiningroomLocation)

    elif fuzz.ratio(outString, 'требования_к_поступлению') >= config['main']['accuracy_comparison'] : showFrame(screenAdmissionRequirements)

    else:
        pass


def findingAudience():
    text = screenEnteringAudienceNumber_body_entryNumber.get()
    if text:
        screenEnteringAudienceNumber_body_entryNumber.delete(0, 'end')
        text = text.replace('А-', '').replace('а-', '')
        text = text.replace('А -', '').replace('а -', '')
        text = text.replace('А- ', '').replace('а- ', '')
        text = text.replace('А - ', '').replace('а - ', '')
        text = text.strip()

        screenAudienceLocation_body_title.config(text = f'Аудитория находится на {text[0]} этаже')
        showFrame(screenAudienceLocation)



root = tkinterTk()

WIDTH = root.winfo_screenwidth()
HEIGHT = root.winfo_screenheight()

root.title('Sign')
root.geometry(f'{WIDTH}x{HEIGHT}')
root.attributes('-fullscreen', True)



screenWelcome = tkinterFrame(root)
screenInfo = tkinterFrame(root)
screenSuportSign = tkinterFrame(root)
screenTest = tkinterFrame(root)
screenMain = tkinterFrame(root)
screenLibraryLocation = tkinterFrame(root)
screenDiningroomLocation = tkinterFrame(root)
screenAccountingLocation = tkinterFrame(root)
screenAdmissionRequirements = tkinterFrame(root)
screenEnteringAudienceNumber = tkinterFrame(root)
screenAudienceLocation = tkinterFrame(root)

# Начальный экран
screenWelcome_appbar = tkinterFrame(screenWelcome, bg='#202020')
screenWelcome_body = tkinterFrame(screenWelcome, bg='#404040')

screenWelcome_appbar.pack(fill='x', side='top')
screenWelcome_body.pack(fill='both', expand=True)

screenWelcome_appbar_informations = tkinterButton(
    screenWelcome_appbar,
    text='Информация',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenInfo)
)

screenWelcome_body_topSpace = tkinterLabel(screenWelcome_body, bg='#404040')
screenWelcome_body_title = tkinterLabel(
    screenWelcome_body,
    text='Здравствуйте!\nЯ могу поговорить с Вами на языке жестов',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)
screenWelcome_body_start = tkinterButton(
    screenWelcome_body,
    text='Начать',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenMain)
)
screenWelcome_body_rumc = tkinterLabel(
    screenWelcome_body,
    text='Вы всегда можете обратиться в ресурсный учебно-методический центр (аудитория А-23)\nВам помогут',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)
screenWelcome_body_bottomSpace = tkinterLabel(screenWelcome_body, bg='#404040')

screenWelcome_appbar_informations.pack(padx=10, pady=10, side='right')
screenWelcome_body_topSpace.pack(fill='both', expand=True)
screenWelcome_body_title.pack()
screenWelcome_body_start.pack(pady=50)
screenWelcome_body_bottomSpace.pack(fill='both', expand=True)
screenWelcome_body_rumc.pack(padx=10, pady=10, side='bottom')


# Экран информации
screenInfo_appbar = tkinterFrame(screenInfo, bg='#202020')
screenInfo_body = tkinterFrame(screenInfo, bg='#404040')

screenInfo_appbar.pack(fill='x', side='top')
screenInfo_body.pack(fill='both', expand=True)

screenWelcome_appbar_back = tkinterButton(
    screenInfo_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenWelcome)
)

screenWelcome_body_suportSign = tkinterButton(
    screenInfo_body,
    text='Слова, которые я понимаю',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenSuportSign)
)
screenWelcome_body_testSign = tkinterButton(
    screenInfo_body,
    text='Проверить распознавание',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenTest)
)

screenWelcome_appbar_back.pack(padx=10, pady=10, side='left')
screenWelcome_body_suportSign.pack(padx=10, pady=10)
screenWelcome_body_testSign.pack(padx=10, pady=10)



# Экран известных жестов
screenSuportSign_appbar = tkinterFrame(screenSuportSign, bg='#202020')
screenSuportSign_body = tkinterFrame(screenSuportSign, bg='#404040')

screenSuportSign_appbar.pack(fill='x', side='top')
screenSuportSign_body.pack(fill='both', expand=True)

screenSuportSign_appbar_back = tkinterButton(
    screenSuportSign_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenInfo)
)

screenSuportSign_body_listSign = tkinterLabel(
    screenSuportSign_body,
    text='     '.join(info['phrases'][:-1]).lower(),
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenSuportSign_appbar_back.pack(padx=10, pady=10, side='left')
screenSuportSign_body_listSign.pack(expand=True)



# Экран тестирования распознавания
screenTest_appbar = tkinterFrame(screenTest, bg='#202020')
screenTest_body = tkinterFrame(screenTest, bg='#404040')

screenTest_appbar.pack(fill='x', side='top')
screenTest_body.pack(fill='both', expand=True)

screenTest_appbar_back = tkinterButton(
    screenTest_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenInfo)
)

screenTest_body_camera = tkinterLabel(
    screenTest_body,
)
screenTest_body_result = tkinterLabel(
    screenTest_body,
    text='Здесь будет результат',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenTest_appbar_back.pack(padx=10, pady=10, side='left')
screenTest_body_camera.pack(padx=10, pady=10, side='left')
screenTest_body_result.pack(padx=10, pady=10, side='right')



# Основной экран
screenMain_appbar = tkinterFrame(screenMain, bg='#202020')
screenMain_body = tkinterFrame(screenMain, bg='#404040')

screenMain_appbar.pack(fill='x', side='top')
screenMain_body.pack(fill='both', expand=True)

screenMain_appbar_back = tkinterButton(
    screenMain_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenWelcome)
)

screenMain_body_title = tkinterLabel(
    screenMain_body,
    text='Начните показывать мне жесты\nЯ постараюсь Вам помочь',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenMain_appbar_back.pack(padx=10, pady=10, side='left')
screenMain_body_title.pack(expand=True)



# Местонахождение библиотеки
screenLibraryLocation_appbar = tkinterFrame(screenLibraryLocation, bg='#202020')
screenLibraryLocation_body = tkinterFrame(screenLibraryLocation, bg='#404040')

screenLibraryLocation_appbar.pack(fill='x', side='top')
screenLibraryLocation_body.pack(fill='both', expand=True)

screenLibraryLocation_appbar_back = tkinterButton(
    screenLibraryLocation_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenMain)
)

screenLibraryLocation_body_title = tkinterLabel(
    screenLibraryLocation_body,
    text='Научная библиотека находится на сайте ЧелГУ\nhttps://library.csu.ru',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenLibraryLocation_appbar_back.pack(padx=10, pady=10, side='left')
screenLibraryLocation_body_title.pack(expand=True)



# Местонахождение столовой
screenDiningroomLocation_appbar = tkinterFrame(screenDiningroomLocation, bg='#202020')
screenDiningroomLocation_body = tkinterFrame(screenDiningroomLocation, bg='#404040')

screenDiningroomLocation_appbar.pack(fill='x', side='top')
screenDiningroomLocation_body.pack(fill='both', expand=True)

screenDiningroomLocation_appbar_back = tkinterButton(
    screenDiningroomLocation_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenMain)
)

screenDiningroomLocation_body_title = tkinterLabel(
    screenDiningroomLocation_body,
    text='Столовая находится на первом этаже театрального корпуса\nКогда Вы зайдёте в корпус, двигайтесь прямо',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenDiningroomLocation_appbar_back.pack(padx=10, pady=10, side='left')
screenDiningroomLocation_body_title.pack(expand=True)



# Местонахождение бухгалтерии
screenAccountingLocation_appbar = tkinterFrame(screenAccountingLocation, bg='#202020')
screenAccountingLocation_body = tkinterFrame(screenAccountingLocation, bg='#404040')

screenAccountingLocation_appbar.pack(fill='x', side='top')
screenAccountingLocation_body.pack(fill='both', expand=True)

screenAccountingLocation_appbar_back = tkinterButton(
    screenAccountingLocation_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenMain)
)

screenAccountingLocation_body_title = tkinterLabel(
    screenAccountingLocation_body,
    text='Бухгалтерия находится в кабинете №300\nВам нужно подняться на третий этаж',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenAccountingLocation_appbar_back.pack(padx=10, pady=10, side='left')
screenAccountingLocation_body_title.pack(expand=True)



# Требования для поспутления
screenAdmissionRequirements_appbar = tkinterFrame(screenAdmissionRequirements, bg='#202020')
screenAdmissionRequirements_body = tkinterFrame(screenAdmissionRequirements, bg='#404040')

screenAdmissionRequirements_appbar.pack(fill='x', side='top')
screenAdmissionRequirements_body.pack(fill='both', expand=True)

screenAdmissionRequirements_appbar_back = tkinterButton(
    screenAdmissionRequirements_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenMain)
)

screenAdmissionRequirements_body_title = tkinterLabel(
    screenAdmissionRequirements_body,
    text='''
Документы, которые необхо​димы для подачи заявления в ЧелГУ

Документ, удостоверяющий личность, гражданство
Документ об образовании
Страховое свидетельство обязательного пенсионного страхования (СНИЛС)
Документы, подтверждающие индивидуальные достижения абитуриента, результаты которых учитываются при приеме на обучение (представляются по усмотрению поступающего)


Подробнее Вы можете узнать на сайте ЧелГУ
https://abit.csu.ru/priem/documents
'''[1:-1],
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenAdmissionRequirements_appbar_back.pack(padx=10, pady=10, side='left')
screenAdmissionRequirements_body_title.pack(expand=True)



# Ввод номера аудитории
screenEnteringAudienceNumber_appbar = tkinterFrame(screenEnteringAudienceNumber, bg='#202020')
screenEnteringAudienceNumber_body = tkinterFrame(screenEnteringAudienceNumber, bg='#404040')

screenEnteringAudienceNumber_appbar.pack(fill='x', side='top')
screenEnteringAudienceNumber_body.pack(fill='both', expand=True)

screenEnteringAudienceNumber_appbar_back = tkinterButton(
    screenEnteringAudienceNumber_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenMain)
)

screenEnteringAudienceNumber_body_title = tkinterLabel(
    screenEnteringAudienceNumber_body,
    text='Введите номер аудитории\n\nНапример\nА-17\nили\n445\n',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)
screenEnteringAudienceNumber_body_entryNumber = tkinterEntry(
    screenEnteringAudienceNumber_body,
    font=('Arial', 14),
    foreground="#FFFFFF",
    background="#606060",
    insertbackground='#FFFFFF',
    bd=0,
    relief='flat'
)
screenEnteringAudienceNumber_body_continue = tkinterButton(
    screenEnteringAudienceNumber_body,
    text='Продолжить',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#404040',
    cursor='hand2',
    bd=0,
    command = lambda: findingAudience()
)

screenEnteringAudienceNumber_appbar_back.pack(padx=10, pady=10, side='left')
screenEnteringAudienceNumber_body_title.pack(padx=10, pady=10)
screenEnteringAudienceNumber_body_entryNumber.pack(side='top', ipady=8)
screenEnteringAudienceNumber_body_continue.pack(padx=10, pady=10, side='top')



# Местонахождение аудитории
screenAudienceLocation_appbar = tkinterFrame(screenAudienceLocation, bg='#202020')
screenAudienceLocation_body = tkinterFrame(screenAudienceLocation, bg='#404040')

screenAudienceLocation_appbar.pack(fill='x', side='top')
screenAudienceLocation_body.pack(fill='both', expand=True)

screenAudienceLocation_appbar_back = tkinterButton(
    screenAudienceLocation_appbar,
    text='Назад',
    font=('Arial', 24),
    foreground="#FFFFFF",
    background='#202020',
    cursor='hand2',
    bd=0,
    command = lambda: showFrame(screenEnteringAudienceNumber)
)

screenAudienceLocation_body_title = tkinterLabel(
    screenAudienceLocation_body,
    text='',
    font=('Arial', 30),
    foreground="#FFFFFF",
    background='#404040',
    justify='center',
    wraplength=WIDTH
)

screenAudienceLocation_appbar_back.pack(padx=10, pady=10, side='left')
screenAudienceLocation_body_title.pack(expand=True)



for frame in (
    screenWelcome, screenInfo, screenSuportSign, screenTest, screenMain,
    screenLibraryLocation, screenDiningroomLocation, screenAccountingLocation,
    screenAdmissionRequirements, screenEnteringAudienceNumber, screenAudienceLocation
    ):
    frame.place(x=0, y=0, width=WIDTH, height=HEIGHT)



showFrame(screenWelcome)
mainLoop()
root.mainloop()