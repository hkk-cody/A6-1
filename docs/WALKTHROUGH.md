# A6-1 단계별 실습 가이드 — NumPy 미니 딥러닝 프레임워크 만들기

이 문서는 빈 폴더에서 시작해 제출까지 가는 순서를 안내합니다. 각 단계는 **목표 / 할 일 / 힌트 / 확인 방법**으로 구성되어 있고, 확인 방법(체크포인트)을 통과한 뒤 다음 단계로 넘어가는 것이 원칙입니다. 완성 코드는 없습니다. 막히면 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)의 해당 장을 먼저 읽습니다.

- 과제 원문: [../subject.md](../subject.md)
- 개념 정리: [LEARNING_GUIDE.md](LEARNING_GUIDE.md)

## 전체 흐름

```mermaid
flowchart LR
    S0["0. 환경과 구조"] --> S1["1~3. Tensor와<br/>자동 미분"]
    S1 --> S4["4. 기울기 검사 함수"]
    S4 --> S5["5~6. 레이어와 초기화"]
    S5 --> S7["7. gradient_check.py"]
    S7 --> S8["8. SGD, Adam"]
    S8 --> S9["9. XOR"]
    S9 --> S10["10. 초기화 실험"]
    S10 --> S11["11. MNIST"]
    S11 --> S12["12. README, 리포트"]
```

과제의 「구현 → 검증 → 실험」 순서를 지킵니다. 검사 함수(4단계)를 레이어보다 먼저 만드는 이유는, 레이어를 하나 만들 때마다 즉시 검증하기 위해서입니다.

## 요구사항 한눈에 보기

| 구분 | 내용 | 출처 |
| --- | --- | --- |
| 구현 언어/라이브러리 | NumPy만 사용. PyTorch, TensorFlow는 로직 참고만 | 7장 제약 사항 |
| 데이터 로더 | `torch.utils.data` 등 자동 로더 금지, 직접 구현 | 7장 |
| 외부 데이터 | UBYTE 또는 CSV 직접 파싱해 NumPy로 로드. 다운로드에 `requests`/`urllib`/`gzip` 허용 | 7장 |
| Python | 3.10 이상 권장 | 6장 |
| 제출 | GitHub 저장소 URL | 2장 |

## 0단계. 환경과 저장소 구조

**목표:** 실행 환경을 만들고, 결과물이 들어갈 자리를 미리 정한다.

**할 일**

1. 가상환경을 만들고 NumPy를 설치합니다. 그래프용으로 Matplotlib을 함께 설치합니다(권장: 그래프 작성은 프레임워크 구현이 아니므로 제약에 걸리지 않지만, 쓴 라이브러리는 README에 밝힙니다).
2. `requirements.txt`를 만듭니다. README 템플릿의 설치 방법이 `pip install -r requirements.txt`입니다.
3. 아래 구조를 참고해 폴더를 만듭니다. `[과제]` 표시는 과제 문서에 직접 등장하는 이름이고, 나머지는 권장 예시입니다.

```text
repo/
├── core/                 # tensor, autograd (README 템플릿의 core)
├── nn/                   # layer, 초기화, 손실 (nn)
├── optim/                # SGD, Adam (optim)
├── data/                 # MNIST 다운로드/파싱, 배치 로더 (권장)
├── gradient_check.py     # [과제] 자가 체크리스트의 스크립트 이름
├── train_xor.py          # 권장 이름
├── train_mnist.py        # 권장 이름
├── init_experiment.py    # 권장 이름
├── figures/              # [과제] 초기화 비교 그래프
├── reports/              # 검증 리포트, 실험 리포트 (권장 위치)
├── README.md             # [과제]
└── requirements.txt      # [과제] README 설치 명령에 등장
```

**힌트:** 결과물 1번은 "tensor, autograd, layer, optimizer 모듈"을 요구합니다. 파일을 `core/tensor.py`, `core/autograd.py`, `nn/layers.py`, `optim/` 식으로 나누면 리뷰어가 대응 관계를 바로 찾을 수 있습니다. MNIST 원본 파일 같은 큰 데이터는 `.gitignore`에 넣고, README에 받는 방법을 적습니다(권장).

**확인 방법**

- [ ] `python --version`이 3.10 이상
- [ ] `python -c "import numpy; print(numpy.__version__)"`가 동작
- [ ] 각 패키지 폴더에서 `import`가 되는지(`__init__.py` 또는 실행 위치 확인)

## 1단계. Tensor 클래스와 backward()

**목표:** `data`, `grad`, `_backward`를 가진 `Tensor`와 위상 정렬 기반 `backward()`를 만든다.

**할 일**

1. `Tensor.__init__`에서 입력을 `float64` NumPy 배열로 저장하고, `grad`를 같은 shape의 0 배열로 둡니다.
2. 재료 노드 목록(`_prev`)과 `_backward`(기본은 아무것도 안 하는 함수)를 둡니다.
3. `backward()`를 구현합니다: 위상 정렬 → 자기 `grad`를 1로 → 역순으로 `_backward()` 호출.
4. `zero_grad()` 또는 그에 해당하는 초기화 수단을 만듭니다.
5. 디버깅용 `__repr__`을 만들어 두면 이후 단계가 편합니다(권장).

**힌트**

```python
class Tensor:
    def __init__(self, data, _children=(), _op=""):
        ...  # data는 np.float64 배열, grad는 zeros_like
    def backward(self):
        ...  # TODO: 위상 정렬 후 역순으로 node._backward()
```

- 위상 정렬은 방문 집합(`set`)과 재귀 함수 하나로 충분합니다. 노드가 수천 개면 재귀 한도에 걸릴 수 있으니, 반복문(스택)으로 바꾸는 방법도 염두에 둡니다.
- 손실이 스칼라가 아닐 때 `backward()`를 부르면 어떻게 할지 정합니다(예: 오류를 내거나, 1로 채운 배열로 시작). 정한 규칙을 README에 적습니다(권장).

**확인 방법:** 아직 연산이 없으므로 다음 단계와 함께 확인합니다.

## 2단계. 사칙연산과 브로드캐스팅 처리

**목표:** `+`, `-`, `*`, `/`의 순전파·역전파를 구현하고, 브로드캐스팅된 기울기를 원래 shape로 되돌린다.

**할 일**

1. `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`와 오른쪽 버전(`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`)을 만듭니다.
2. 숫자가 들어오면 `Tensor`로 감싸는 처리를 넣습니다.
3. 기울기를 대상 shape로 되돌리는 도우미 함수 `unbroadcast(grad, shape)`를 만들고, 모든 연산의 `_backward`에서 사용합니다.
4. 기울기는 항상 `+=`로 누적합니다.

**힌트:** 공식은 [LEARNING_GUIDE.md 5장](LEARNING_GUIDE.md#5-연산별-역전파-공식)의 표를, 축을 줄이는 규칙은 [2장](LEARNING_GUIDE.md#2-numpy-배열-shape-브로드캐스팅)을 봅니다.

```python
def unbroadcast(grad, shape):
    # TODO 1: grad.ndim > len(shape)이면 앞쪽 축을 sum으로 제거
    # TODO 2: shape에서 크기 1인 축은 sum(axis=i, keepdims=True)
    return grad
```

**확인 방법 (체크포인트)**

- [ ] $x=2, y=-3, z=4$, $f = x \cdot y + z$에서 `f.data == -2`, `x.grad == -3`, `y.grad == 2`, `z.grad == 1`
- [ ] $f = x \cdot x$ ($x = 3$)에서 `x.grad == 6` (누적 확인)
- [ ] $a=6, b=2$, $f = a / b$에서 `a.grad == 0.5`, `b.grad == -1.5`
- [ ] `X`가 `(2, 3)`, `b`가 `(3,)`일 때 `(X + b).sum()`의 `b.grad`가 `[2, 2, 2]`이고 shape `(3,)`
- [ ] `zero_grad` 없이 `backward()`를 두 번 부르면 기울기가 누적되어 값이 달라지는 것을 확인(중간 노드 `q`의 `grad`도 남아 있으므로 `x.grad`는 정확히 두 배가 아니라 그보다 커질 수 있음). 그래프의 모든 노드 `grad`를 0으로 만든 뒤(또는 순전파를 새로 한 뒤) 다시 부르면 원래 값으로 돌아오는지 확인

## 3단계. 행렬 곱과 보조 연산

**목표:** MatMul과 레이어·손실에 필요한 연산을 완성한다.

**할 일**

1. `__matmul__`(`@`)의 순전파와 역전파를 구현합니다.
2. 필요에 따라 `sum(axis, keepdims)`, `mean`, `exp`, `log`, `__pow__`, `reshape` 등을 추가합니다.
3. 각 연산의 결과 `grad`의 shape가 입력 `data`의 shape와 같은지 확인하는 `assert`를 개발 중에 넣어 둡니다(권장).

**힌트:** 행렬 곱 공식은 shape로 복원합니다. `(N, D) @ (D, H)`에서 `dA`는 `(N, D)`, `dB`는 `(D, H)`가 되도록 `g`, `A.T`, `B.T`를 조합합니다. `sum`의 역전파는 "줄어든 축으로 기울기를 다시 펼치는" 작업이며 `np.broadcast_to`나 `keepdims`가 유용합니다.

**확인 방법**

- [ ] `A` `(5, 3)`, `B` `(3, 2)`에 대해 `(A @ B).sum().backward()` 후 `A.grad.shape == (5, 3)`, `B.grad.shape == (3, 2)`
- [ ] 위 결과에서 `A.grad`의 모든 행이 `B.sum(axis=1)`과 같은지 손으로 검산

## 4단계. 기울기 검사 함수 만들기

**목표:** 수치 기울기와 상대 오차를 계산하는 함수를 먼저 만들어, 이후 모든 연산과 레이어를 즉시 검증한다.

**할 일**

1. 스칼라를 돌려주는 함수 `f`와 검사할 배열을 받아, 원소마다 중심 차분으로 수치 기울기를 구하는 함수를 만듭니다. $\varepsilon = 10^{-5}$ (과제 지정).
2. 해석적 기울기와 수치 기울기의 상대 오차를 구하는 함수를 만듭니다.
3. 2~3단계의 모든 연산(`+`, `-`, `*`, `/`, `@`, 그리고 추가한 보조 연산)에 대해 검사를 돌립니다.

**힌트**

```python
def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    # TODO: np.nditer 등으로 원소 하나씩 x[i] += eps / -= eps 후 원상복구
    return grad

def relative_error(a, n):
    ...  # TODO: LEARNING_GUIDE 7장의 정의, 분모 0 처리 규칙 포함
```

- 원소 값을 바꾼 뒤 **반드시 원래 값으로 되돌립니다**. 되돌리지 않으면 다음 원소 검사가 틀어집니다.
- `f`는 매번 새로 순전파를 하는 함수여야 합니다. 이전 그래프의 `grad`가 섞이지 않도록 합니다.
- 비교 대상의 손실은 $L = \sum(\text{out} \cdot R)$ ($R$은 고정된 무작위 배열)처럼 만들면 모든 출력 원소가 기울기에 기여합니다.

**확인 방법**

- [ ] 모든 기본 연산의 상대 오차가 1e-7 이하
- [ ] 일부러 곱셈 역전파의 부호를 바꿔 보면 상대 오차가 크게 튀는지(검사기가 버그를 잡는지) 확인 후 원상복구

## 5단계. 레이어 모듈 (Linear, ReLU, Sigmoid, Softmax)과 손실

**목표:** 과제가 요구하는 네 레이어를 모듈로 만들고, 학습에 쓸 손실 함수를 준비한다.

**할 일**

1. 공통 부모 클래스(예: `Module`)를 두고 `__call__`/`forward`, `parameters()`를 정의합니다(권장 설계).
2. `Linear(in_features, out_features, init=...)`: 파라미터 `W`, `b`를 `Tensor`로 보관합니다. 초기화 옵션 자리는 6단계에서 채웁니다.
3. `ReLU`, `Sigmoid`, `Softmax`를 구현합니다. 기존 연산의 조합으로 만들어도 되고, 자체 순전파·역전파 식을 가진 연산으로 만들어도 됩니다.
4. 손실 함수를 만듭니다. 이진 분류(XOR)는 Sigmoid + 이진 교차 엔트로피 또는 2클래스 Softmax + 교차 엔트로피, MNIST는 Softmax + 교차 엔트로피가 일반적입니다.
5. 여러 레이어를 순서대로 실행하는 컨테이너(예: `Sequential`)를 만들면 XOR/MNIST 스크립트가 짧아집니다(권장).

**힌트**

- Softmax는 행마다 최댓값을 빼서 안정화합니다(`axis=-1`, `keepdims=True`).
- `log(0)`을 막으려면 확률을 아주 작은 값으로 `clip`하거나, Softmax와 교차 엔트로피를 합친 기울기 $(p - y)/N$를 씁니다. 합친 버전을 쓰더라도 Softmax 레이어 자체의 Gradient Check는 따로 해야 합니다.
- Sigmoid는 큰 음수 입력에서 overflow 경고가 나므로 입력 범위를 제한합니다.

**확인 방법**

- [ ] 4단계 검사기로 각 레이어의 입력 기울기와 Linear의 `W`, `b` 기울기가 1e-7 이하
- [ ] `Softmax` 출력의 각 행 합이 1
- [ ] $z = [2, 1, 0]$의 Softmax가 약 `[0.665, 0.245, 0.090]`, 정답 0번일 때 교차 엔트로피 약 0.408
- [ ] ReLU 검사 시 입력에 0 근처 값이 없도록 했는지

## 6단계. 가중치 초기화 옵션 4종

**목표:** Zero, Random, He, Xavier 초기화를 옵션으로 제공한다.

**할 일**

1. 초기화 함수들을 한 파일에 모읍니다(예: `nn/init.py`).
2. `Linear`가 `init="zero" | "random" | "he" | "xavier"` 같은 인자로 방식을 고르게 합니다.
3. 난수는 시드를 받을 수 있게 합니다(`np.random.default_rng(seed)`). 실험 재현에 필요합니다.

**힌트:** 과제 정의를 그대로 따릅니다.

| 옵션 | 정의 (과제 원문) |
| --- | --- |
| Zero | 모든 가중치 0 |
| Random | 표준정규분포 |
| He | $N(0, \sqrt{2/n_{in}})$ — 두 번째 인자를 표준편차로 사용 |
| Xavier | $N(0, \sqrt{2/(n_{in}+n_{out})})$ |

`randn`에 **표준편차**를 곱합니다. 분산을 곱하는 실수를 조심합니다. 편향은 0으로 두는 것이 일반적입니다.

**확인 방법**

- [ ] `Linear(784, 128, init="he")`의 `W.data.std()`가 약 0.0505
- [ ] `Linear(128, 64, init="xavier")`의 `W.data.std()`가 약 0.102
- [ ] `init="zero"`면 `W.data`가 전부 0, `init="random"`이면 표준편차 약 1
- [ ] 같은 시드로 두 번 만들면 같은 가중치

## 7단계. gradient_check.py 완성과 검증 리포트

**목표:** "모든 레이어의 오차가 1e-7 이하"를 한 번의 실행으로 보여 주는 스크립트와 실행 로그를 만든다.

**할 일**

1. 프로젝트 루트에 `gradient_check.py`를 만듭니다(자가 체크리스트에 나오는 이름).
2. 기본 연산(사칙연산, MatMul)과 레이어(Linear, ReLU, Sigmoid, Softmax)를 각각 검사하고 결과를 한 줄씩 출력합니다.
3. 마지막 줄에 최대 상대 오차와 통과 여부를 출력합니다.
4. 실행 결과를 파일로 저장해 검증 리포트에 넣습니다. 예: `python gradient_check.py | tee reports/gradient_check.log`

**힌트:** 출력 형식은 [../subject.md](../subject.md) 8장 예시를 참고하되 자유롭게 정합니다.

```text
[Pass] Layer Linear_1 Gradient Check - Analytic: ..., Numerical: ..., Diff: ...
All gradient checks passed! Maximum difference: ... < 1e-07
```

- 검사 입력은 고정 시드로 만들어 매번 같은 결과가 나오게 합니다.
- 실패 항목이 있으면 0이 아닌 종료 코드로 끝내면 자동 검사에 쓰기 좋습니다(권장).
- 검증 리포트에는 로그와 함께 사용한 상대 오차 정의, $\varepsilon$, dtype을 적습니다.

**확인 방법**

- [ ] 모든 줄이 `[Pass]`이고 최대 오차가 1e-7 이하
- [ ] 로그 파일이 저장소에 있음

## 8단계. 옵티마이저 SGD와 Adam

**목표:** 파라미터 목록을 받아 `step()`과 `zero_grad()`를 제공하는 옵티마이저 두 개를 만든다.

**할 일**

1. `SGD(params, lr)`: $\theta \leftarrow \theta - \eta g$
2. `Adam(params, lr, betas, eps)`: 파라미터마다 `m`, `v` 상태와 스텝 카운터 `t`, 편향 보정 포함
3. 업데이트는 `param.data`를 직접 바꿉니다. 업데이트 자체가 계산 그래프에 기록되면 안 됩니다.

**힌트:** 식은 [LEARNING_GUIDE.md 9장](LEARNING_GUIDE.md#9-옵티마이저)에 있습니다. Adam 첫 스텝의 이동량은 기울기 크기와 상관없이 약 `lr`입니다.

**확인 방법**

- [ ] $f(w) = (w-3)^2$를 $w = 0$에서 출발해 SGD(`lr=0.1`)로 100스텝 돌리면 $w \approx 3$
- [ ] 같은 문제를 Adam(`lr=0.1`, 수백 스텝)으로 풀어도 3 근처로 수렴 (Adam은 스텝당 이동량이 약 `lr`이라 SGD보다 스텝이 더 필요할 수 있음)
- [ ] Adam 첫 스텝 후 $w$의 변화량이 `lr`과 거의 같음

## 9단계. XOR 학습 (He 초기화)

**목표:** He 초기화로 100 에폭 안에 Loss < 0.1 또는 정확도 95% 이상을 달성한다.

**할 일**

1. XOR 데이터 4개를 NumPy 배열로 만듭니다.
2. 은닉층이 있는 작은 MLP(예: 2 → 은닉 → 출력)를 He 초기화로 만듭니다.
3. 에폭마다 순전파 → 손실 → `zero_grad` → `backward` → `step`을 하고 손실을 기록합니다.
4. 마지막에 예측값과 정확도를 출력합니다.

**힌트**

- 4개를 한 번에 넣으면 1 에폭 = 1 업데이트입니다. 100번 안에 수렴하려면 Adam이나 충분한 학습률이 필요할 수 있습니다. 에폭 정의(배치 크기)를 리포트에 적습니다.
- ReLU 은닉 뉴런이 너무 적으면 죽은 뉴런 때문에 시드에 따라 실패할 수 있습니다. 은닉 크기를 조금 늘려 봅니다.
- 정확도 95% 이상은 4개 중 4개 정답을 뜻합니다.

**확인 방법**

- [ ] 100 에폭 이내에 Loss < 0.1 또는 정확도 100% (He Init)
- [ ] 여러 시드에서도 대체로 성공하는지(권장)

## 10단계. 초기화 비교 실험 (Zero vs Random vs He)

**목표:** 초기화 방식에 따른 Loss 감소 속도 차이를 그래프와 표로 보이고, Zero Init 실패를 재현한다.

**할 일**

1. 9단계 XOR 설정에서 **초기화만** 바꿔 Zero / Random / He를 각각 학습합니다. Xavier 포함은 선택입니다.
2. 에폭별 손실을 저장하고, 세 곡선을 한 그래프에 겹쳐 `figures/`에 PNG로 저장합니다(X축 Epoch, Y축 Loss).
3. 특정 에폭(예: 5, 50)의 손실과 수렴 여부를 표로 정리합니다.
4. 실험 리포트에 관찰 → 원인 → 근거 순으로 결론을 씁니다.

**힌트**

- Zero Init은 50 에폭 후에도 손실이 줄지 않아야 합니다(약 0.6931 유지). 줄어든다면 Zero를 가중치에만 적용했는지, 다른 곳에 무작위성이 섞였는지 확인합니다.
- XOR 입력층은 $n_{in} = 2$라 He와 Random의 표준편차가 같습니다. 차이가 작게 나오면 그 이유를 분석에 쓰고, 은닉층을 넓히거나 층을 늘린 추가 실험을 해 볼 수 있습니다(권장).
- 그래프에는 축 이름, 범례, 제목을 넣습니다.

**확인 방법**

- [ ] `figures/`에 세 곡선이 겹친 그래프 파일이 있음
- [ ] Zero Init 곡선이 50 에폭 동안 평평함
- [ ] 리포트에 표와 원인 분석(대칭성, 분산)이 있음

## 11단계. MNIST 로더와 1 에폭 학습

**목표:** MNIST 바이너리를 직접 파싱하고, 직접 만든 배치 로더로 1 에폭 학습해 정확도 80% 이상을 확인한다.

**할 일**

1. MNIST IDX 파일(학습/테스트 이미지와 레이블 4개)을 내려받습니다. `urllib`/`requests`와 `gzip` 사용은 허용됩니다. 원본 배포처 접근이 막혀 있을 수 있으므로 공개 미러나 CSV 버전을 찾아 쓰고, 출처를 README에 적습니다.
2. 헤더를 빅엔디언으로 읽어 매직 넘버(이미지 2051, 레이블 2049)와 개수, 크기를 확인한 뒤 나머지를 `uint8` 배열로 읽습니다.
3. `(N, 784)`로 펼치고 0~1로 정규화, 레이블은 원-핫으로 바꿉니다.
4. 섞인 인덱스로 미니배치를 내주는 로더를 직접 만듭니다.
5. MLP(예: 784 → 은닉 → 10, He 초기화, Softmax + 교차 엔트로피)를 1 에폭 학습하고 테스트 정확도를 출력합니다.

**힌트**

```python
header = np.frombuffer(raw[:16], dtype=">i4")   # magic, n, rows, cols
pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
```

- 학습 중 일정 스텝마다 손실을 출력해 내려가는지 봅니다.
- 1 에폭이 너무 느리면 배치 크기를 조절합니다. 순수 NumPy라도 작은 MLP는 수 분 안에 끝나는 것이 보통입니다.

**확인 방법**

- [ ] 매직 넘버가 2051/2049, 학습 이미지 shape `(60000, 784)`, 테스트 `(10000, 784)`
- [ ] 첫 이미지를 28×28로 그려 보면 숫자 모양이고 레이블과 일치
- [ ] 1 에폭 후 테스트 정확도 80% 이상

## 12단계. README와 리포트 정리

**목표:** 다른 사람이 저장소만 보고 설치, 실행, 결과 확인을 할 수 있게 한다.

**할 일:** [../subject.md](../subject.md) 8장의 README 필수 섹션 템플릿을 따릅니다.

| README 섹션 | 들어갈 내용 |
| --- | --- |
| 프로젝트 개요 | 프레임워크 목적과 구조 |
| 설치 방법 | `pip install -r requirements.txt` |
| 실행 방법 | 각 스크립트 실행 명령과 순서 (gradient_check → XOR → 초기화 실험 → MNIST 등) |
| 모듈 구조 | core / nn / optim 간 의존 관계 (mermaid 다이어그램 활용 가능) |
| 사용 예시 | Tensor 생성 → 연산 → `backward` 호출 예시 코드 |
| 실험 결과 요약 | Gradient Check 통과 여부, 초기화 실험 결론 |

검증 리포트(실행 로그)와 실험 리포트(그래프와 분석)를 README에서 링크합니다.

**확인 방법**

- [ ] 새 가상환경에서 README의 명령만으로 모든 스크립트가 실행됨
- [ ] README의 사용 예시 코드를 복사해 실행하면 동작함

## 자주 겪는 문제

| 증상 | 흔한 원인 | 확인할 곳 |
| --- | --- | --- |
| 상대 오차가 1e-2 근처 | `float32` 사용 | `Tensor` 생성 시 dtype |
| 상대 오차가 특정 레이어만 크게 나옴 | 해당 연산의 역전파 식 오류, unbroadcast 누락 | 2·3단계 체크포인트 |
| ReLU 검사만 가끔 실패 | 0 근처 입력 | 검사 입력 생성 방식 |
| Softmax 검사에서 두 기울기가 모두 0 | 손실을 `sum(out)`으로 정의 | 가중합 손실 사용 |
| `W -= lr * grad`에서 shape가 바뀜 | `grad` shape가 `data`와 다름 | unbroadcast |
| 손실이 에폭마다 계속 증가 | `zero_grad` 누락으로 기울기 누적 | 학습 루프 |
| 손실이 `nan` | 학습률 과다, `log(0)`, `exp` overflow | clip, 안정화 Softmax |
| Zero Init인데 손실이 줄어듦 | 일부 층만 0, 또는 다른 무작위 요소 | 초기화 적용 범위 |
| MNIST 개수가 이상한 큰 수 | 리틀엔디언으로 헤더를 읽음 | dtype `>i4` |
| MNIST 정확도 10% 근처 | 레이블과 이미지 순서 불일치, 정규화 누락 | 첫 이미지와 레이블 비교 |

## 보너스 과제 (선택)

필수 항목을 모두 끝낸 뒤 시도합니다. 힌트만 제공합니다.

**1. Dropout과 L2 정규화**

- Inverted Dropout: 학습 모드에서 마스크를 만들어 곱하고 $1/(1-p)$로 키웁니다. 평가 모드에서는 입력을 그대로 통과시킵니다. 레이어에 모드 전환 수단(예: `train()`/`eval()`)이 필요합니다.
- 마스크는 순전파 때 만든 것을 역전파에서도 그대로 써야 합니다.
- L2(Weight Decay): 옵티마이저에서 기울기에 $\lambda\theta$를 더합니다. 효과를 보이려면 과적합이 생기는 설정(적은 학습 데이터, 큰 모델)에서 학습/테스트 정확도 차이를 비교합니다.
- Gradient Check 때는 Dropout을 끄거나 마스크를 고정합니다.

**2. 추가 활성화 함수**

- Tanh, LeakyReLU, ELU의 미분은 [LEARNING_GUIDE.md 12장](LEARNING_GUIDE.md#12-보너스-개념-선택-과제용)의 표에 있습니다. 미분을 손으로 유도해 리포트에 적고, `gradient_check.py`에 검사 항목을 추가합니다.
- LeakyReLU와 ELU도 0에서 꺾이므로 ReLU와 같은 입력 주의가 필요합니다.

## 최종 자가 점검표

| subject.md 요구사항 | 어디서 확인하는지 |
| --- | --- |
| Tensor에 `data`, `grad`, `_backward` 속성 | `core/` 소스, 1단계 |
| `backward`가 연쇄 법칙으로 기울기 전파 | 2단계 체크포인트 ($f = xy + z$) |
| `+`, `-`, `*`, `/`, MatMul 순전파·역전파 | 2·3단계, `gradient_check.py` 로그 |
| 수치 기울기 vs 해석적 기울기 비교 함수 | 4단계 검사 함수 |
| 중심 차분, $\varepsilon = 10^{-5}$ | 검사 함수 코드, 검증 리포트 |
| 상대 오차 1e-7 이하 증명 테스트 (모든 레이어) | `gradient_check.py` 실행 로그 |
| Linear, ReLU, Sigmoid, Softmax 모듈 | `nn/` 소스, 5단계 |
| Zero / Random / He / Xavier 초기화 옵션 | `nn/` 초기화 코드, 6단계 체크포인트 |
| SGD, Adam | `optim/` 소스, 8단계 |
| XOR: He Init, 100 에폭 내 Loss < 0.1 또는 정확도 95% 이상 | XOR 스크립트 출력 |
| MNIST 분류 코드, 1 에폭 정확도 80% 이상 | MNIST 스크립트 출력 |
| MNIST 바이너리/CSV 직접 파싱, 자동 로더 미사용 | `data/` 로더 코드 |
| Zero Init 실패 재현 (50 에폭 후 Loss 감소 없음) | 초기화 실험 그래프와 표 |
| 초기화별 Loss 비교 그래프가 `figures/`에 있음 | `figures/` 폴더 |
| 검증 리포트 (실행 로그) | `reports/` 등 로그 파일 |
| 실험 리포트 (그래프와 분석) | 실험 리포트 문서 |
| README: 개요, 설치, 실행, 모듈 구조, 사용 예시, 결과 요약 | `README.md`, 12단계 |
| 구현은 NumPy만 사용 | `requirements.txt`, import 문 |
| GitHub 저장소 URL 제출 | 원격 저장소에 모든 파일 push 여부 |
