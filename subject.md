# A6-1 — AI의 속 엔진(두뇌)을 내 손으로 직접 만들어보기

출처: [Codyssey · AI 심화 학습](https://usr.codyssey.kr/learning/learningMap?projectNo=136002)
확인일: 2026-10-31

> Codyssey 과제 안내를 Markdown으로 옮긴 문서입니다. 요구사항과 참고 예시는 원문을 따릅니다.

- 분야: AI/SW 심화 · 구분: 딥러닝 · 학습시간: 80시간 · 필수/선택: 필수
- 난이도: ★★★

## 1. 미션 소개

딥러닝 모델의 학습 실패 원인은 대부분 보이지 않는 수학적 오류나 초기 설정의 미세한 차이에서 기인한다. 이 미션은 NumPy를 활용해 자동 미분(AutoGrad) 엔진을 구현하는 것에서 나아가, 기울기 검증(Gradient Checking)과 가중치 초기화(Initialization) 실험을 통해 견고한 학습 엔진을 구축하는 미션이다. 학습자는 역전파 수식을 코드로 구현하는 것에 그치지 않고, 수치 미분과의 비교를 통해 구현의 정확성을 증명해야 하며, 초기화 전략이 학습 수렴에 미치는 영향을 실험을 통해 파악한다. 이 과정을 통해 딥러닝 프레임워크의 내부 동작을 깊이 이해하고, 학습 실패 시 원인을 진단할 수 있는 역량을 갖추게 된다.

미션의 학습 흐름은 「구현 → 검증 → 실험」의 3단계로 진행된다. 먼저 Tensor와 AutoGrad 엔진을 구현한 뒤, Gradient Checking으로 역전파 로직의 무결성을 진단하고, 초기화 실험으로 학습 실패 원인을 재현-분석한다. 이 서사를 통해 학습자는 ‘왜 이 순서로 해야 하는지’를 체감하게 된다

## 2. 최종 결과물

다음 내용이 모두 포함된 GitHub 레포지토리(Repository) URL을 제출한다.

1. 미니 딥러닝 프레임워크 소스 코드 (tensor, autograd, layer, optimizer 모듈)
2. 검증 리포트(Verification Report) : 수치 미분과 역전파 결과의 오차를 검증한 스크립트 실행 로그
3. 실험 리포트(Experiment Report) : 초기화 방법(Zero vs Random vs He)에 따른 Loss 감소 비교 그래프 및 분석
4. README.md : 프레임워크 구조 설명, 설치 및 실행 방법, 모듈별 사용 예시 포함
5. 학습 검증 스크립트 : XOR 문제 해결 및 MNIST 분류 수행 코드

- 미션 완료 자가 체크리스트
  - gradient_check.py 실행 시 모든 레이어의 오차가 1e-7 이하인가?
  - Zero Init으로 학습 실패를 재현할 수 있는가?
  - He Init으로 XOR 문제가 정상 동작하는가? (Loss < 0.1 또는 정확도 95% 이상)
  - 초기화 비교 그래프가 figures/ 폴더에 있는가?
  - README.md 에 설치 및 실행 방법이 명시되어 있는가?
  - MNIST 분류 시 1 Epoch에서 정확도 80% 이상인가?

## 3. 과제 목표

이 과제를 마친 후, 학습자는 아래를 스스로 설명할 수 있어야 한다.

- 계산 그래프(Computational Graph)의 동적 생성 및 순전파/역전파(Backpropagation) 원리를 설명할 수 있다.
- 수치 미분(Numerical Differentiation)을 활용하여 역전파 로직의 무결성을 검증하는 원리를 이해한다.
- 가중치 초기화(Initialization)가 기울기 소실(Vanishing Gradient) 및 폭주(Exploding Gradient)에 미치는 영향을 설명할 수 있다.
- 연쇄 법칙(Chain Rule)이 코드 레벨에서 어떻게 구현되는지 이해한다.
- 텐서의 브로드캐스팅(Broadcasting)이 기울기 계산에 미치는 영향을 설명할 수 있다

## 4. 기능 요구 사항

다음 요구사항을 모두 만족해야 한다.

1. 텐서 및 자동 미분 엔진
   - Tensor 클래스는 데이터(data), 기울기(grad), 생성 연산 정보(_backward)를 속성으로 가져야 한다.
   - backward 메서드는 연결된 노드들에 연쇄 법칙(Chain Rule)을 통해 기울기를 전파해야 한다.
   - 기본 사칙연산(+, -, *, /) 및 행렬 곱(MatMul)에 대한 순전파와 역전파 로직을 구현해야 한다.
2. 무결성 검증 시스템 (Gradient Checking)
   - 수식으로 계산된 기울기(Analytic Gradient)와 미소 변동을 이용한 수치 기울기(Numerical Gradient)의 차이를 계산하는 함수를 구현한다.
   - 수치 미분 공식: grad ≈ [f(x+ε) - f(x-ε)] / (2ε), ε = 1e-5
   - 두 기울기의 상대 오차(Relative Difference)가 1e-7 이하임을 증명하는 테스트 코드를 포함해야 한다.
3. 레이어 및 초기화 전략
   - Linear, ReLU, Sigmoid, Softmax 레이어를 모듈화하여 구현한다.
   - 다음 가중치 초기화 방식을 옵션으로 제공해야 한다:
   - Zero Initialization: 모든 가중치를 0으로 초기화
   - Random Initialization: 표준정규분포에서 샘플링
   - He Initialization: N(0, √(2/n_in))에서 샘플링 (ReLU용)
   - Xavier Initialization: N(0, √(2/(n_in+n_out)))에서 샘플링 (Sigmoid/Tanh용)
     - Xavier는 구현 필수, 초기화 비교 실험에서의 포함은 선택이다. 필수 비교 실험은 Zero/Random/He 3종으로 수행한다.
4. 학습 및 최적화
   - SGD(Stochastic Gradient Descent) 및 Adam 옵티마이저를 구현한다.
   - XOR 문제 해결 및 MNIST 분류를 수행하여 프레임워크의 정상 동작을 검증한다.
   - 초기화 방식에 따라 학습 초기의 Loss 감소 속도가 달라지는 것을 그래프로 시각화한다.

- 성공 판정 최소 기준
  | **검증 항목** | **필수 기준** | **비고** |
  | --- | --- | --- |
  | Gradient Ceck | 모든 레이어 오차 < 1e-7 | 상대 오차 기준 |
  | XOR 문제 (필수) | 100 Epoch 내 Loss < 0.1 또는 정확도 95% 이상 | He Init 기준 |
  | MNIST 분류 (선택) | 1 Epoch 정확도 80% 이상 | 소형 실험 기준 |
  | Zero Init 실패 재현 | 50 Epoch 후 Loss 감소 없음 | 대칭성 문제 확인 |

## 5. 보너스 과제 (선택)

1. Dropout 및 Regularization
   - 학습 시와 테스트 시의 동작이 다른 Dropout 레이어(Inverted Dropout 방식)를 구현한다.
   - L2 Regularization(Weight Decay)을 옵티마이저에 추가하여 과적합 억제 효과를 확인한다.
   - 배움 포인트 : 정규화 기법이 모델의 일반화 성능에 미치는 영향을 직접 확인한다
2. 다양한 활성화 함수 구현
   - Tanh, LeakyReLU, ELU 등 추가 활성화 함수를 구현하고 역전파 로직을 추가한다.
   - 배움 포인트 : 다양한 수식의 미분 과정을 직접 유도해보며 수학적 이해도를 높인다

### [참고 사항] 데이터셋 확보 가이드

미션 배포 시 별첨 문서나 `Appendix` 형태로 제공할 수 있는 가이드 내용입니다.

1. 라이브러리 내장 데이터셋 (Standard Datasets)
   - 활용 목적: 알고리즘 검증, 벤치마킹, 초기 프로토타이핑
   - DL 1(프레임워크 구현) 미션에서 학습 로직의 정상 동작 여부를 빠르게 확인하기 위해 권장합니다.
   - 방법:
     - `Scikit-learn`: `load_digits()` (MNIST 소형판), `make_moons()` (비선형 분류용)
     - `Torchvision`/`Keras`: MNIST, CIFAR-10, Fashion-MNIST
   - 주의사항: 데이터 로드 과정이 너무 쉬워 전처리의 중요성을 놓칠 수 있으므로, DL 1에서는 데이터를 `NumPy` 배열로 변환하여 사용하는 과정을 반드시 거치도록 합니다.
2. 오픈 데이터 플랫폼 (Open Repositories)
   - 활용 목적: 특정 도메인(의료, 제조, 커머스 등) 데이터 확보
   - DL 2(전이 학습) 미션에서 다양한 클래스나 실제 이미지를 구하기 위해 활용합니다.
   - 추천 사이트:
     - Kaggle Datasets: 가장 방대한 소스. (예: `Dog vs Cat`, `Food-101`)
     - Hugging Face Datasets: NLP 및 이미지 데이터 접근이 용이함.
     - AI Hub (한국지능정보사회진흥원): 한국형 데이터(K-Food, 한국어 텍스트) 확보 시 유용.
     - UCI Machine Learning Repository: 정형 데이터 및 시계열 데이터.
3. API 활용 (Real-time Data APIs)
   - 활용 목적: 시계열 데이터, 금융, 날씨 등 연속성 있는 데이터 확보
   - DL 2(시계열 예측) 미션에서 가장 현업과 유사한 데이터 파이프라인을 구축하는 방법입니다.
   - 추천 라이브러리:
     - Finance: `yfinance` (Yahoo Finance 주가 데이터, 별도 가입 없이 사용 가능하여 강력 추천)
     - Economy: `Fredapi` (미국 연방준비은행 경제 데이터)
     - Crypto: `pyupbit` (업비트 암호화폐 거래 데이터)
4. 데이터 증강 (Data Augmentation) & 합성 (Synthetic)
   - 활용 목적: 데이터 부족 상황(Few-shot) 해결
   - DL 2(전이 학습) 미션의 핵심인 **'데이터가 부족할 때의 전략'**을 위해 필수적입니다.
   - 방법:
     - 이미지: `Albumentations`, `ImgAug` 라이브러리를 사용하여 회전, 자르기, 노이즈 추가.
     - 시계열: `Jittering`(노이즈 추가), `Scaling`(크기 변경), `Time-warping` 등으로 데이터 뻥튀기.

## 6. 개발 환경

- Python 3.10 이상 권장

## 7. 제약 사항

- PyTorch, TensorFlow 등 딥러닝 전용 라이브러리는 로직 참고용으로만 확인하고, 실제 구현에는 NumPy만 사용한다.
- 데이터 로더 구현 시 torch.utils.data 등 자동 로더를 사용하지 않고 직접 구현한다.
- 외부 데이터셋(MNIST 등) 사용 시 바이너리(UBYTE) 또는 CSV를 직접 파싱하여 NumPy로 로드한다. (다운로드에는 requests/urllib/gzip 사용 허용)

## 8. 결과 예시

아래는 정답이 아니라 참고 예시다. 실제 문구와 구조는 달라도 된다.

- 검증 로그 예시:
  ```text
  [Pass] Layer Linear_1 Gradient Check - Analytic: 0.0342, Numerical: 0.0342, Diff: 2.3e-08
  [Pass] Layer Linear_2 Gradient Check - Analytic: -0.0156, Numerical: -0.0156, Diff: 1.8e-09
  All gradient checks passed! Maximum difference: 2.3e-08 < 1e-07 ✓
  ```
- 초기화 실험 결과 예시:
  | **Initialization** | **Epoch 5 Loss** | **Epoch 50 Loss** | **Converged?** |
  | --- | --- | --- | --- |
  | Zero Init | 0.6931 | 0.6931 | No (Stuck) |
  | Random Init | 0.4521 | 0.0892 | Yes (Slow) |
  | He Init | 0.2134 | 0.0123 | Yes (Fast) |
- 실험 결론 : Zero Initialization은 대칭성(Symmetry) 문제로 모든 뉴런이 동일한 값을 학습하여 Loss가 감소하지 않았다. Random Initialization은 분산이 크게 설정되어 초반 학습이 불안정했으나, He Initialization은 ReLU 활성화 함수에 최적화된 분산으로 5 Epoch 만에 빠르게 수렴했다
- Loss 감소 그래프 예시 :  
  X축: Epoch (0-100), Y축: Loss (0-1.0) 세 가지 초기화 방식의 Loss 곡선을 하나의 그래프에 오버레이하여 비교한다
- README.md 필수 섹션 템플릿
  - 프로젝트 개요 : 프레임워크 목적과 구조 설명
  - 설치 방법 : 의존성 설치 명령어 (pip install -r requirements.txt)
  - 실행 방법 : 각 스크립트 실행 커맨드 및 순서
  - 모듈 구조 : core/nn/optim 간 의존 관계 설명
  - 사용 예시 : Tensor 생성 → 연산 → backward 호출 예시 코드
  - 실험 결과 요약 : Gradient Check 통과 여부, 초기화 실험 결론
