import sys
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 폰트 관리를 위해 추가

# 코랩 환경을 위한 기본 경로 설정
PROJECT = Path('/content')
ASSETS = PROJECT / 'assets'
ASSETS.mkdir(exist_ok=True)
MAX_BYTES = 1 * 1024**3

FILES = {
    'resnet50-transfer.pth': 'https://raw.githubusercontent.com/AthanasiosNathanail/Fossil-Image-Classification-with-Pytorch/main/resnet50-transfer.pth',
    'sample.jpg': 'https://raw.githubusercontent.com/AthanasiosNathanail/Fossil-Image-Classification-with-Pytorch/main/sample.jpg',
}

# ----------------------------------------------------
# 한글 폰트 설정 (런타임 재시작 없이 즉시 적용)
# ----------------------------------------------------
FONT_URL = 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
FONT_PATH = ASSETS / 'NanumGothic.ttf'

if not FONT_PATH.exists():
    urlretrieve(FONT_URL, FONT_PATH)

# 다운로드한 폰트를 matplotlib에 등록하고 기본 폰트로 설정합니다.
fe = fm.FontEntry(fname=str(FONT_PATH), name='NanumGothic')
fm.fontManager.ttflist.insert(0, fe)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# ----------------------------------------------------
# 전역(Global) 변수 초기화
# ----------------------------------------------------
device = None
model = None
preprocess = None
class_names = []

def download_if_missing(filename, url):
    target = ASSETS / filename
    if target.exists():
        print(f'이미 있음: {target.name} ({target.stat().st_size / 1024**2:.1f} MiB)')
        return target
    with urlopen(Request(url, method='HEAD')) as response:
        size = int(response.headers.get('Content-Length', 0))
    assert 0 < size <= MAX_BYTES, f'다운로드하지 않음: {size / 1024**2:.1f} MiB'
    urlretrieve(url, target)
    print(f'다운로드 완료: {target.name} ({target.stat().st_size / 1024**2:.1f} MiB)')
    return target

def prepare_class():
    WEIGHTS_PATH = download_if_missing('resnet50-transfer.pth', FILES['resnet50-transfer.pth'])
    SAMPLE_PATH = download_if_missing('sample.jpg', FILES['sample.jpg'])
    return WEIGHTS_PATH, SAMPLE_PATH

######################################################
def AI_model_load(WPATH):
    # 전역 변수 사용 선언
    global device, model, preprocess, class_names

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 다운로드한 가중치 파일을 불러옵니다.
    checkpoint = torch.load(WPATH, map_location='cpu', weights_only=False)
    class_names = [checkpoint['idx_to_class'][i] for i in range(len(checkpoint['idx_to_class']))]

    model = models.resnet50(weights=None)
    model.fc = checkpoint['fc']             # 원본 프로젝트의 6-클래스 분류기
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()

    # ImageNet 전처리: 긴 변을 줄이고 중앙 224×224 영역을 사용한 뒤 색상 범위를 표준화합니다.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('실행 장치:', device)
    print('분류 후보:', ', '.join(class_names))  

######################################################
def classify_fossil(photo_file, top_k=3):
    # 전역 변수 사용 선언
    global device, model, preprocess, class_names, PROJECT

    photo_path = PROJECT / photo_file
    
    if model is None or preprocess is None:
        raise RuntimeError("모델이 로드되지 않았습니다. 먼저 AI_model_load()를 실행해 주세요.")

    image = Image.open(photo_path).convert('RGB')
    
    batch = preprocess(image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        probabilities = model(batch).exp()[0].cpu()
    values, indices = probabilities.topk(top_k)
    
    result = pd.DataFrame({
        '순위': range(1, top_k + 1),
        '후보': [class_names[i] for i in indices.tolist()],
        '확률': values.tolist(),
    })
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'입력: {Path(photo_path).name}')
    plt.show()
    return result.style.format({'확률': '{:.1%}'})