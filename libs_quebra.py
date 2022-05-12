from keras.models import load_model
from helpers import resize_to_fit
import cv2
import pickle
import numpy as np
import os
from PIL import Image

def tratar_imagens(arquivo):
    imagem = cv2.imread(arquivo)
    imagem_cinza = cv2.medianBlur(imagem, 3) 
    ret, imagem_tradada = cv2.threshold(imagem_cinza, 211, 255, cv2.THRESH_BINARY )
    cv2.imwrite("temp.png", imagem_tradada)
    return(quebrar_captcha("temp.png"))

def quebrar_captcha(arquivo):
    with open("rotulos_modelo.dat","rb") as arquivo_tradutor:
        lb = pickle.load(arquivo_tradutor)
    modelo = load_model("modelo_treinado.hdf5")
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regiao_letras = []
    for contorno in contornos:
        (x, y, l, a) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area > 30:
            if (area > 500):
                regiao_letras.append((x, y, l-round(l/2), a))
                regiao_letras.append((x+round(l/2), y, l-round(l/2), a))
            else:
                regiao_letras.append((x, y, l, a))
    regiao_letras = sorted(regiao_letras, key = lambda i: i[0])
    imagem_final = cv2.merge([imagem] * 3)
    previsao=[]
    for retangulo in regiao_letras:
        x, y, l, a = retangulo
        imagem_letra = imagem[y:y+a, x:x+l]
        imagem_letra = resize_to_fit(imagem_letra, 20, 20)
        imagem_letra = np.expand_dims(imagem_letra, axis=2)
        imagem_letra = np.expand_dims(imagem_letra, axis=0)
        letra_prevista = modelo.predict(imagem_letra)
        letra_prevista = lb.inverse_transform(letra_prevista)[0]
        previsao.append(letra_prevista)
        nome_arquivo = os.path.basename(arquivo)
        cv2.rectangle(imagem_final, (x,y), (x+l, y+a), (0,255,0),1)
    return ("".join(previsao))