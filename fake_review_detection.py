### Input : 리뷰 내용, 사용자 ID를 입력받음
### Output : 리뷰내용 검증을 통해 정상 리뷰 판정이 되면 skip, 가짜리뷰면 리뷰내용과 id를 output으로 보냄
### 추가 설명 : 기존에 데이터를 array로 저장한 파일을 통하여 sqlite DB파일을 생성, 그 후 db파일 내 데이터를 통해 가짜리뷰 예측 진행함.
### 추가적으로 db파일 내 black_list 테이블에 가짜 리뷰로 식별된 유저의 유사도 점수, ID, 리뷰 내용을 저장함.

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import sqlite3
import io
import datetime
import os

# print(os.getcwd())
sentencebert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

def cosine(a,b):
  return np.dot(a,b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())   

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def main():
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array) # 이거 두 줄이 중요함. 이거없으면 배열 불러올때 오류걸려서 개같음

    con = sqlite3.connect("./db.sqlite3", detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()

    cur.execute("drop table black_list")
    cur.execute("drop table train_vector")
    cur.execute("create table if not exists train_vector (arr array)")
    cur.execute("create table if not exists black_list (score text, userid text, review text, datetime datetime, product_id text)") # blacklist 저장할 테이블 생성
    # cur.execute('insert into train_vector (arr) VALUES(?)',(train_vector,))

    cur.execute("select arr from train_vector")
    data = cur.fetchone()[0]
    print(data.shape)


    review = str(input("리뷰 입력 : "))
    ID = str(input("ID 입력 : "))
    product_id = 2

    review_ebd = sentencebert_model.encode(review)

    score = 0
    for i in range(len(data)):
        if score <= cosine(review_ebd, data[i,:]):
            score = cosine(review_ebd, data[i,:])
    if score >= 0.95 and len(review) >= 10:
        print("해당 리뷰는 가짜리뷰 일 수 있습니다!")
        cur.execute('insert into black_list VALUES(?,?,?,?,?);', (str(score), ID, review, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(product_id)))
        # cur.execute("select * from black_list")
        # data = cur.fetchone()
        # print(data)
        # db blacklist 테이블에 유사도 점수, 유저ID, 리뷰 내용 추가
    else:
        print("해당 리뷰는 정상 리뷰로 분류됩니다!")
    
    data = np.append(data, review_ebd.reshape(1,-1), axis = 0)
    cur.execute('DELETE FROM "train_vector"')
    cur.execute('insert into train_vector (arr) VALUES(?)',(data,))

    con.commit() # db 변경요소 저장
    cur.close()  #   Cursor
    con.close()
main()