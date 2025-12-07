# Color Tiles 강화학습 프로젝트

Color Tiles 게임을 플레이하는 강화학습 AI를 학습하고 실행하는 프로젝트입니다.

## 프로젝트 개요

- **게임**: Color Tiles (23×15 보드, 10가지 색상, 200개 타일)
- **알고리즘**: PPO (Proximal Policy Optimization)
- **프레임워크**: Stable-Baselines3 + Gymnasium
- **GUI**: PyQt6

## 설치 방법

### 1. 기본 의존성 설치

```bash
# Python 3.8 이상 필요
pip install -e .
```

### 2. 강화학습 의존성 설치

```bash
pip install gymnasium stable-baselines3 torch tensorboard
```

또는:

```bash
pip install -r requirements-rl.txt
```

## 사용 방법

### 1. GUI로 게임 플레이 (사람)

```bash
python main.py
```

기본 게임 플레이:
1. "게임 시작" 버튼 클릭
2. 빈 셀을 클릭하여 타일 제거
3. 120초 내에 모든 타일 제거 시 승리!

---

### 2. AI 학습

#### 짧은 테스트 학습 (10K steps)

```bash
python -m rl.training.train --total-timesteps 10000 --n-envs 2
```

이 명령어는:
- 10,000 timesteps 동안 학습
- 2개의 병렬 환경 사용
- 약 5-10분 소요 (CPU 기준)
- 체크포인트를 `checkpoints/` 디렉토리에 저장

#### 본격 학습 (1M steps)

```bash
python -m rl.training.train --total-timesteps 1000000 --n-envs 8
```

이 명령어는:
- 1,000,000 timesteps 동안 학습
- 8개의 병렬 환경 사용
- 약 10-20시간 소요 (CPU 기준)
- 매 10,000 steps마다 체크포인트 저장
- 매 5,000 steps마다 평가 수행

#### 학습 파라미터

```bash
python -m rl.training.train \
  --total-timesteps 1000000 \
  --n-envs 8 \
  --learning-rate 3e-4 \
  --seed 42 \
  --save-dir checkpoints
```

**파라미터 설명:**
- `--total-timesteps`: 총 학습 스텝 수 (기본값: 1,000,000)
- `--n-envs`: 병렬 환경 개수 (기본값: 8)
- `--learning-rate`: 학습률 (기본값: 3e-4)
- `--seed`: 랜덤 시드 (기본값: 0)
- `--save-dir`: 체크포인트 저장 디렉토리 (기본값: checkpoints)

#### 학습 재개 (체크포인트에서)

```bash
python -m rl.training.train \
  --checkpoint checkpoints/ppo_colortiles_step_50000.zip \
  --total-timesteps 1000000
```

---

### 3. 학습 모니터링 (TensorBoard)

```bash
tensorboard --logdir logs/tensorboard/
```

그런 다음 브라우저에서 http://localhost:6006 접속

**확인 가능한 지표:**
- Episode reward (에피소드 보상)
- Win rate (승리 비율)
- Mean tiles cleared (평균 제거 타일 수)
- Episode length (에피소드 길이)
- Policy loss, Value loss
- Entropy (탐험 정도)

---

### 4. GUI에서 학습된 AI 플레이

#### Step 1: GUI 실행

```bash
python main.py
```

#### Step 2: AI 설정

1. **체크포인트 선택**:
   - 우측 "AI 플레이어" 패널에서 체크포인트 드롭다운 클릭
   - 학습된 모델 선택 (예: `ppo_colortiles_best.zip`)
   - "새로고침" 버튼으로 목록 갱신 가능

2. **AI 시작**:
   - "AI 시작" 버튼 클릭
   - 게임이 자동 시작되고 AI가 플레이 시작

3. **속도 조절**:
   - 속도 슬라이더로 1-10 조절 (초당 액션 수)
   - 1: 느림 (관찰 용이)
   - 10: 빠름

4. **AI 중지**:
   - "중지" 버튼으로 언제든지 중지 가능

#### AI 상태 정보

GUI에서 다음 정보 확인 가능:
- **스텝**: 현재 에피소드의 스텝 수
- **가치 추정**: AI가 예측한 state value
- **행동 신뢰도**: 선택한 액션의 확률
- **다음 행동**: AI가 선택할 위치 (row, col)
- **하이라이트**: 보드에서 다음 액션 위치를 색상으로 표시
  - 🟢 녹색: 높은 신뢰도 (>80%)
  - 🟡 노란색: 중간 신뢰도 (50-80%)
  - 🟠 주황색: 낮은 신뢰도 (<50%)

---

## 프로젝트 구조

```
color-tiles-rl/
├── src/
│   ├── color_tiles/          # 게임 엔진
│   │   ├── domain/           # 도메인 모델 (Color, Position, GameState)
│   │   ├── engine/           # 게임 로직 (Board, GameEngine)
│   │   ├── gui/              # PyQt6 GUI
│   │   │   ├── main_window.py
│   │   │   ├── board_widget.py
│   │   │   ├── ai_control_panel.py   # AI 제어 패널
│   │   │   └── ai_status_panel.py    # AI 상태 표시
│   │   └── utils/            # 유틸리티 (BoardGenerator)
│   └── rl/                   # 강화학습 모듈
│       ├── env/
│       │   └── color_tiles_env.py     # Gymnasium 환경
│       ├── training/
│       │   ├── train.py               # 학습 스크립트
│       │   └── callbacks.py           # 커스텀 콜백
│       └── inference/
│           └── ai_player.py           # AI 플레이어
├── tests/
│   └── test_color_tiles_env.py        # 환경 테스트
├── checkpoints/              # 학습된 모델 저장 (자동 생성)
├── logs/                     # TensorBoard 로그 (자동 생성)
├── docs/
│   └── reinforce_learning_plan.md     # RL 계획서
├── main.py                   # GUI 실행 파일
├── README.md
└── pyproject.toml
```

---

## 강화학습 환경 스펙

### State (관찰 공간)

- **타입**: `Box(0, 10, (15, 23), int8)`
- **형태**: 15×23 2D 그리드
- **값**:
  - 0: 빈 셀
  - 1-10: 색상 (Color enum value + 1)

### Action (행동 공간)

- **타입**: `Discrete(345)`
- **범위**: 0-344 (23×15 = 345개 셀)
- **변환**:
  - `row = action // 23`
  - `col = action % 23`

### Reward (보상)

| 상황 | 보상 |
|------|------|
| 타일 제거 | `+1.0 × 타일 수` |
| 무효 이동 | `-10.0` |
| 승리 | `+100.0` |
| 패배 | `-(남은 타일 × 2)` |

### Episode 종료

- **Terminated**: 승리 (모든 타일 제거) 또는 패배 (시간 초과/막힘)
- **Truncated**: Max steps (200) 도달

---

## PPO 하이퍼파라미터

```python
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}
```

---

## 예상 학습 진행

| 단계 | Steps | Win Rate | 평균 타일 제거 | Invalid Move Rate |
|------|-------|----------|----------------|-------------------|
| 초기 | 0-50K | 0-5% | 30-50 | 60-80% |
| 초중기 | 50K-200K | 5-30% | 80-120 | 30-50% |
| 중기 | 200K-500K | 30-60% | 140-170 | 15-25% |
| 후기 | 500K-1M | 60-80% | 175-195 | 5-10% |
| 수렴 | 1M+ | 80%+ | 195+ | <5% |

---

## 체크포인트 관리

학습 중 자동으로 다음 체크포인트가 생성됩니다:

```
checkpoints/
├── ppo_colortiles_step_10000.zip    # 10K steps
├── ppo_colortiles_step_20000.zip    # 20K steps
├── ...
├── ppo_colortiles_best.zip          # 최고 성능 모델
└── ppo_colortiles_final.zip         # 최종 모델
```

**권장 사항:**
- `best.zip`: 평가 성능이 가장 좋은 모델 (GUI에서 사용 추천)
- `final.zip`: 학습 완료 후 최종 모델
- `step_*.zip`: 특정 시점의 모델 (학습 재개 시 사용)

---

## 테스트

### 환경 테스트

```bash
pytest tests/test_color_tiles_env.py -v
```

### AI Player 테스트

```bash
python -m rl.inference.ai_player checkpoints/ppo_colortiles_best.zip
```

---

## 트러블슈팅

### 1. ModuleNotFoundError: No module named 'rl'

**문제**: Python이 `rl` 패키지를 찾지 못함

**해결**:
```bash
# 프로젝트 루트에서
pip install -e .
```

### 2. stable-baselines3 not installed

**문제**: RL 라이브러리가 설치되지 않음

**해결**:
```bash
pip install gymnasium stable-baselines3 torch tensorboard
```

### 3. CUDA out of memory (GPU 사용 시)

**문제**: GPU 메모리 부족

**해결**:
```bash
# CPU 사용 강제
export CUDA_VISIBLE_DEVICES=""
python -m rl.training.train ...
```

### 4. GUI에서 체크포인트가 보이지 않음

**문제**: `checkpoints/` 디렉토리에 파일이 없음

**해결**:
1. 먼저 학습을 실행하여 체크포인트 생성
2. GUI에서 "새로고침" 버튼 클릭

---

## 성능 최적화

### CPU 학습 가속

```bash
# 병렬 환경 수 증가 (CPU 코어 수에 맞게)
python -m rl.training.train --n-envs 16
```

### GPU 사용

```bash
# PyTorch가 자동으로 GPU 감지
# device="auto"로 설정되어 있음
python -m rl.training.train --total-timesteps 1000000
```

---

## 참고 문서

- **RL 계획서**: `docs/reinforce_learning_plan.md`
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **PPO 논문**: https://arxiv.org/abs/1707.06347

---

## 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 작성자

- Color Tiles 게임 엔진: jmlee
- 강화학습 통합: Claude (Anthropic)

---

## 버전 히스토리

- **v1.0.0** (2025-12-07): 초기 릴리스
  - PPO 기반 RL 환경 구현
  - 학습 파이프라인 구현
  - GUI 통합 완료
