# BiLSTM을 이용한 영화 리뷰 감성분석 ver2
**Embedding layer**을 통해 임베딩 된 네이버 영화 리뷰 데이터를 **BiLSTM**을 통해 긍정, 부정을 분류해 주는 프로젝트

## 1. 모델 구조도
![alt text](https://github.com/MSWon/Sentimental-Analysis-ver2/blob/master/images/model.png "Model")

1. 정답이 있는 네이버 영화 리뷰 데이터 15만건([박은정님 제공](https://github.com/e9t/nsmc))에 대해서 **품사 태깅**

2. 품사 태깅한 단어들에 대해 **Embedding layer**을 이용해 임베딩 벡터로 변환

3. 단어 벡터들을 **BiLSTM**에 넣어서 양쪽 끝 state들에 대해서 **fully connected layer**와 **Softmax**함수를 이용해 분류

## 2. 필요한 패키지

- [konlpy](http://konlpy.org/en/v0.4.4/)
- [tensorflow >= 1.12.0](https://www.tensorflow.org/)


## 3. 데이터

- Training data : 영화 리뷰 데이터 15만건 [ratings_train.txt](https://github.com/e9t/nsmc)

- Test data : 영화 리뷰 데이터 5만건 [ratings_test.txt](https://github.com/e9t/nsmc)

## 4. 학습

**1. Git clone**
```
$ git clone https://github.com/MSWon/Sentimental-Analysis-ver2.git
```
**2. Training with user settings**
```
$ python train.py --batch_size 128 --word_dim 512 --hidden_dim 512 --num_layers 2 --training_epochs 5
```

## 5. 결과

- Test accuracy : **85.32%**
- Doc2Vec, Term-existance Naive Bayes에 의한 성능 보다 뛰어남([박은정](https://www.slideshare.net/lucypark/nltk-gensim))
- test_example.py를 통해 직접 입력한 문장에 성능 확인

Pre-trained 모델 다운로드
```
$ sh download_model.sh
```
직접 문장을 입력하여 테스트
```
$ python test_example.py

문장을 입력하세요: 재밌던데!
긍정입니다

문장을 입력하세요: 진짜 보자마자 잠듬 .. ㅋㅋ
부정입니다

문장을 입력하세요: 배우진 좋고 스토리도 좋았다
긍정입니다

문장을 입력하세요: 애매한 스토리
부정입니다
```
