# 🕵️‍♂️AI-engineers
- 데이터 사이언티스트 면접 질문 및 답변
- 질문 리스트 : [zzsza](https://github.com/zzsza)님 git hub참고,[WeareSoft](https://github.com/WeareSoft/tech-interview)님 참고

## 네트워크
- 네트워크 vs 네트워킹
- OSI 7계층이란
- TCP/IP 4계층이란
- IPv4주소의 기본 구조
- Encapulation과 Decapulation이란?
- 서브넷 마스크와 CIDR란?
- TCP와 UDP의 차이?
- TCP의 3-way-handshake와 4-way-handsake
- TCP의 연결 설정 과정과 연결 종료 과정이 차이가 나는 이유는?
- HTTP 메서드 종류와 설명
- 쿠키 vs 세션
- DNS란?
- REST와 RESTful의 개념과 차이점
- MAC adress란?
- www.naver.com에 접속했을때, 요청을 보내고 받기까지의 과정을 설명
- URI와 URL, URN
- WAS란 무엇인가요?
- TCP/IP 송수신 구조에 대해 간략히 설명


## 머신러닝

- Cross Validation은 무엇이고 어떻게 해야하나요?
- 회귀 / 분류시 알맞은 metric은 무엇일까요?
- 알고 있는 metric에 대해 설명해주세요(ex. RMSE, MAE, recall, precision ...)
- 정규화를 왜 해야할까요? 정규화의 방법은 무엇이 있나요?
- Local Minima와 Global Minima에 대해 설명해주세요.
- 차원의 저주에 대해 설명해주세요
- dimension reduction기법으로 보통 어떤 것들이 있나요?
- PCA는 차원 축소 기법이면서, 데이터 압축 기법이기도 하고, 노이즈 제거기법이기도 합니다. 왜 그런지 설명해주실 수 있나요?
- LSA, LDA, SVD 등의 약자들이 어떤 뜻이고 서로 어떤 관계를 가지는지 설명할 수 있나요?
- Markov Chain을 고등학생에게 설명하려면 어떤 방식이 제일 좋을까요?
- 텍스트 더미에서 주제를 추출해야 합니다. 어떤 방식으로 접근해 나가시겠나요?
- SVM은 왜 반대로 차원을 확장시키는 방식으로 동작할까요? 거기서 어떤 장점이 발생했나요?
- 다른 좋은 머신 러닝 대비, 오래된 기법인 나이브 베이즈(naive bayes)의 장점을 옹호해보세요.
- Association Rule의 Support, Confidence, Lift에 대해 설명해주세요.
- 최적화 기법중 Newton’s Method와 Gradient Descent 방법에 대해 알고 있나요?
- 머신러닝(machine)적 접근방법과 통계(statistics)적 접근방법의 둘간에 차이에 대한 견해가 있나요?
- 인공신경망(deep learning이전의 전통적인)이 가지는 일반적인 문제점은 무엇일까요?
- 지금 나오고 있는 deep learning 계열의 혁신의 근간은 무엇이라고 생각하시나요?
- ROC 커브에 대해 설명해주실 수 있으신가요?
- 여러분이 서버를 100대 가지고 있습니다. 이때 인공신경망보다 Random Forest를 써야하는 이유는 뭘까요?
- K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)
- L1, L2 정규화에 대해 설명해주세요
- XGBoost을 아시나요? 왜 이 모델이 캐글에서 유명할까요?
- 앙상블 방법엔 어떤 것들이 있나요?
- SVM은 왜 좋을까요?
- feature vector란 무엇일까요?
- 좋은 모델의 정의는 무엇일까요?
- 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?
- 스팸 필터에 로지스틱 리그레션을 많이 사용하는 이유는 무엇일까요?
- OLS(ordinary least squre) regression의 공식은 무엇인가요?


## 딥러닝 일반

- 딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?
- 왜 갑자기 딥러닝이 부흥했을까요?
- 마지막으로 읽은 논문은 무엇인가요? 설명해주세요
- Cost Function과 Activation Function은 무엇인가요?
- Tensorflow, Keras, PyTorch, Caffe, Mxnet 중 선호하는 프레임워크와 그 이유는 무엇인가요?
- Data Normalization은 무엇이고 왜 필요한가요?
- 알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등)
- 오버피팅일 경우 어떻게 대처해야 할까요?
- 하이퍼 파라미터는 무엇인가요?
- Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요?
- 볼츠만 머신은 무엇인가요?
- 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?
- Non-Linearity라는 말의 의미와 그 필요성은?
- ReLU로 어떻게 곡선 함수를 근사하나?
- ReLU의 문제점은?
- Bias는 왜 있는걸까?
- Gradient Descent에 대해서 쉽게 설명한다면?
- 왜 꼭 Gradient를 써야 할까? 그 그래프에서 가로축과 세로축 각각은 무엇인가? 실제 상황에서는 그 그래프가 어떻게 그려질까?
- GD 중에 때때로 Loss가 증가하는 이유는?
- 중학생이 이해할 수 있게 더 쉽게 설명 한다면?
- Back Propagation에 대해서 쉽게 설명 한다면?
- Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?
- GD가 Local Minima 문제를 피하는 방법은?
- 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?
- Training 세트와 Test 세트를 분리하는 이유는?
- Validation 세트가 따로 있는 이유는?
- Test 세트가 오염되었다는 말의 뜻은?
- Regularization이란 무엇인가?
- Batch Normalization의 효과는?
- Dropout의 효과는?
- BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?
- GAN에서 Generator 쪽에도 BN을 적용해도 될까?
- SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?
- SGD에서 Stochastic의 의미는?
- 미니배치를 작게 할때의 장단점은?
- 모멘텀의 수식을 적어 본다면?
- 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?
- 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
- Back Propagation은 몇줄인가?
- CNN으로 바꾼다면 얼마나 추가될까?
- 간단한 MNIST 분류기를 TF, Keras, PyTorch 등으로 작성하는데 몇시간이 필요한가?
- CNN이 아닌 MLP로 해도 잘 될까?
- 마지막 레이어 부분에 대해서 설명 한다면?
- 학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?
- 만약 한글 (인쇄물) OCR을 만든다면 데이터 수집은 어떻게 할 수 있을까?
- 딥러닝할 때 GPU를 쓰면 좋은 이유는?
- 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?
- GPU를 두개 다 쓰고 싶다. 방법은?
- 학습시 필요한 GPU 메모리는 어떻게 계산하는가?
- TF, Keras, PyTorch 등을 사용할 때 디버깅 노하우는?
- 뉴럴넷의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가?

## NLP
- One Hot 인코딩에 대해 설명해주세요
- POS 태깅은 무엇인가요? 가장 간단하게 POS tagger를 만드는 방법은 무엇일까요?
- 문장에서 “Apple”이란 단어가 과일인지 회사인지 식별하는 모델을 어떻게 훈련시킬 수 있을까요?
- 영어 텍스트를 다른 언어로 번역할 시스템을 어떻게 구축해야 할까요?
- 뉴스 기사를 주제별로 자동 분류하는 시스템을 어떻게 구축할까요?
- Stop Words는 무엇일까요? 이것을 왜 제거해야 하나요?
- 영화 리뷰가 긍정적인지 부정적인지 예측하기 위해 모델을 어떻게 설계하시겠나요?
- TF-IDF 점수는 무엇이며 어떤 경우 유용한가요?
- Regular grammar는 무엇인가요? regular expression과 무슨 차이가 있나요?
- RNN에 대해 설명해주세요
- LSTM은 왜 유용한가요?
- Translate 과정 Flow에 대해 설명해주세요
- n-gram은 무엇일까요?
- PageRank 알고리즘은 어떻게 작동하나요?
- depedency parsing란 무엇인가요?
- Word2Vec의 원리는?
- bert와 gpt의 차이?


## 컴퓨터 비전
- 딥러닝 발달 이전에 사물을 Detect할 때 자주 사용하던 방법은 무엇인가요?
- Faster R-CNN의 장점과 단점은 무엇인가요?
- dlib는 무엇인가요?
- YOLO의 특징
- Object Detection 알고리즘의 종류와 특징
- Residual Network는 왜 잘 될까요? Ensemble과 관련이 있을까요?
- CAM(Class Activation Map)는 무엇인가요?
- 자율주행 자동차의 원리는 무엇인가요?
- Semantic Segmentation은 무엇인가요?
- Visual Q&A는 무엇인가요?
- Image Captiong은 무엇인가요?
- Fully Connected Layer의 기능은 무엇인가요?
- Neural Style는 어떻게 진행될까요?
- CNN이 MLP보다 좋은 이유는?
- 어떤 CNN의 파라미터 개수를 게산해 본다면?
- 시퀀스 데이터에 CNN을 적용하는 것이 가능할까?

