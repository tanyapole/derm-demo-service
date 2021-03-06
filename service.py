from flask import Flask, send_file, jsonify
import flask as F
from PIL import Image

import torch
from pathlib import Path
import model_creating
import torchvision.transforms as TF
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn

app = Flask(__name__)
device = torch.device('cuda:1')

primary = np.array(['пятно+эритема', 'бугорок', 'узел', 'папула+бляшка+комедон', 'волдырь', 'пузырек', 'пузырь', 'гнойничок'])
use_primary = np.array([0,2,3,5,6,7])

diseases = np.array(['атопический дерматит', 'акне', 'псориаз', 'розацеа', 'бородавки', 'герпес', 'красный полский лишай', 'витилиго', 'аллергический контактный дерматит', 'экзема', 'дерматомикозы', 'буллезный пемфигоид', 'пузырчатка', 'контагиозный моллск', 'крапивница', 'себорейный кератоз', 'чесотка', 'себорейный дерматит', 'актинический кератоз', 'базалиома'])

def load_model(fldr, num_classes):
    m = model_creating.create_model(num_classes)
    src_fldr = Path('../derm-dis-morph')
    pt = list((src_fldr/fldr).iterdir())[0]
    m.load_state_dict(torch.load(pt))
    m = m.to(device).eval()
    return m

def load_img(b_stream):
    img = Image.open(b_stream)
    _normalize = TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _to_tensor = TF.ToTensor()
    img = cv2.resize(np.array(img), (224,224))
    img = _normalize(_to_tensor(img))
    return img.unsqueeze(dim=0).to(device)

def _np(t): return t.detach().cpu().numpy()

print('Loading models...')
ds_morph_fldrs = [
    'logs/weights/run_20210614_173615',
    'logs/weights/run_20210614_154853',
    'logs/weights/run_20210614_141023',
    'logs/weights/run_20210614_122605',
    'logs/weights/run_20210614_104527',
]
ds_morph_models = [load_model(fldr, len(use_primary)) for fldr in ds_morph_fldrs]

ds_dis_fldrs = [
    'logs/weights/run_20210614_134931',
    'logs/weights/run_20210614_122929',
    'logs/weights/run_20210614_083858',
    'logs/weights/run_20210614_083507',
    'logs/weights/run_20210613_233526',
]
ds_dis_models = [load_model(fldr, len(diseases)) for fldr in ds_dis_fldrs]

cv_dis_fldrs = [
    'logs/weights/run_20210614_134732',
    'logs/weights/run_20210614_083901',
    'logs/weights/run_20210613_233015',
    'logs/weights/run_20210613_232242',
    'logs/weights/run_20210613_231954',
]
cv_dis_models = [load_model(fldr, len(diseases)) for fldr in cv_dis_fldrs]
print('Models loaded')

@app.route('/')
def hello_world():
    # test any response
    print('[log] recieved query')
    return 'Hello, World!'

@app.route("/im_size", methods=["POST"])
def process_image():
    # test sending images
    file = F.request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})

@app.route('/predict', methods=['POST'])
def predict():
    # test sending images
    file = F.request.files['image']
    # Read the image via file.stream
    img = load_img(file.stream)
    # inference
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad(): 
        morph_probs = [_np(sigmoid(m(img))) for m in ds_morph_models]
        probs = [_np(softmax(m(img))) for m in ds_dis_models]
    morph_expected_probs = get_expected(morph_probs)[0]
    expected_probs = get_expected(probs)
    # convert prediction to human readable
    pred_morph = primary[np.array(use_primary)[morph_expected_probs > 0.5]].tolist()
    ensemble_pred = np.argmax(expected_probs, axis=-1)[0]
    pred_disease = diseases[ensemble_pred].item()

    return jsonify({'msg': 'success',
                   'disease': pred_disease,
                   'morphology': pred_morph})

def get_entropy(probs): return np.sum(-probs*np.log(probs), axis=1)
def get_expected(L): return np.stack(L, axis=0).mean(axis=0)

@app.route('/uncertainty', methods=['POST'])
def uncertainty():
    # test sending images
    file = F.request.files['image']
    # Read the image via file.stream
    img = load_img(file.stream)
    # inference
    softmax = nn.Softmax(dim=1)
    with torch.no_grad(): 
        probs = [_np(softmax(m(img))) for m in cv_dis_models]
    # compute uncertainty
    expected_probs = get_expected(probs)
    entropy_of_expected = get_entropy(expected_probs)
    expected_entropy = get_expected([get_entropy(_) for _ in probs])

    totalU = entropy_of_expected
    dataU = expected_entropy
    knowU = totalU - dataU
    
    # ensemble prediction
    ensemble_pred = np.argmax(expected_probs, axis=-1)[0]
    pred_disease = diseases[ensemble_pred].item()

    return jsonify({'msg': 'success',
                   'disease': pred_disease,
                    'data_uncertainty': dataU.item(),
                    'knowledge_uncertainty': knowU.item(),
                   'total_uncertainty': totalU.item()})



app.run(port=9000, threaded=False, host="127.0.0.1")


