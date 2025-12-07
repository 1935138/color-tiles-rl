# Color Tiles 강화학습 계획서

## 1. 개요

Color Tiles 게임에 강화학습을 적용하여 AI가 자동으로 게임을 플레이하도록 합니다.

### 1.1 게임 특성 요약
- **보드 크기**: 23 x 15 (345개 셀)
- **타일 수**: 200개 (10가지 색상 x 20개씩)
- **빈 셀 수**: 145개
- **행동**: 빈 셀을 클릭하여 4방향의 같은 색 타일 2개 이상 제거
- **승리 조건**: 120초 내 모든 타일 제거
- **페널티**: 무효한 이동 시 10초 감소

---

## 2. 강화학습 알고리즘 선택

### 2.1 후보 알고리즘 비교

| 알고리즘 | 장점 | 단점 | 적합도 |
|---------|------|------|--------|
| **DQN** | 이산 행동 공간에 적합, 구현 단순 | 학습 불안정, 하이퍼파라미터 민감 | ★★★☆☆ |
| **Double DQN** | DQN의 과대평가 문제 해결 | 여전히 experience replay 필요 | ★★★★☆ |
| **PPO** | 안정적 학습, 튜닝 용이, 범용성 | 샘플 효율성 낮음 | ★★★★★ |
| **A2C** | 병렬 처리 가능, 빠른 학습 | PPO보다 불안정 | ★★★☆☆ |

### 2.2 선택: PPO (Proximal Policy Optimization)

**선택 이유:**

1. **안정적인 학습**: PPO는 policy gradient 업데이트를 clipping하여 급격한 정책 변화를 방지합니다. 이는 게임의 복잡한 상태 공간에서 안정적인 학습을 보장합니다.

2. **Exploration-Exploitation 균형**: Entropy bonus를 통해 탐험을 장려하면서도 안정적으로 수렴합니다. Penalty를 통해 유효한 행동을 자연스럽게 학습할 수 있습니다.

3. **하이퍼파라미터 튜닝 용이**: PPO는 다른 알고리즘에 비해 하이퍼파라미터에 덜 민감하여 초기 설정으로도 좋은 결과를 얻을 수 있습니다.

4. **검증된 성능**: Atari, 보드게임 등 다양한 환경에서 검증된 알고리즘입니다.

5. **구현 라이브러리**: Stable-Baselines3에서 잘 구현되어 있어 빠른 프로토타이핑이 가능합니다.

---

## 3. 환경 설계 (Gymnasium Interface)

### 3.0 RL 기본 구조

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Interaction Loop                     │
│                                                             │
│      ┌───────┐                              ┌─────────────┐│
│      │       │          State (s_t)         │             ││
│      │       │◄─────────────────────────────│             ││
│      │ Agent │                              │ Environment ││
│      │       │──────────────────────────────►│             ││
│      │       │         Action (a_t)         │             ││
│      └───────┘                              │             ││
│          ▲                                  │             ││
│          │          Reward (r_t)            │             ││
│          └──────────────────────────────────│             ││
│                                             └─────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Color Tiles RL 매핑:**
- **Agent**: PPO 신경망 (Actor-Critic)
- **Environment**: ColorTilesEnv (23×15 보드 게임)
- **State**: 보드 상태 (타일 배치, 남은 시간, 남은 타일 수)
- **Action**: 셀 선택 (345개 중 하나)
- **Reward**: 타일 제거 점수 또는 무효 이동 페널티

---

### 3.1 강화학습 환경 설계 요약

#### 3.1.1 핵심 RL 구성 요소

| RL 구성 요소 | 설명 | 세부 사항 |
|----------|------|----------|
| **Agent** | PPO Policy Network | Actor-Critic 구조의 신경망 |
| **Environment** | ColorTilesEnv | Gymnasium 기반 Color Tiles 게임 환경 |
| **State** | Box(0, 10, (15, 23), int8) 또는<br>Box(0, 1, (15, 23, 11), float32) | **옵션 1**: 2D 그리드 (0=빈셀, 1-10=색상)<br>**옵션 2**: One-hot 인코딩 (11채널)<br>**추가 정보**: 남은 시간, 남은 타일 수 |
| **Action** | Discrete(345) | 23×15=345개 셀 위치<br>action → (row, col) 변환 필요 |
| **Reward** | 타일 제거 기반 | **타일 제거**: +1.0 × 타일 수<br>**무효 이동**: -10.0<br>**승리**: +100.0<br>**패배**: -(남은 타일 × 10) |

#### 3.1.2 Episode 설정

| 설정 항목 | 값 | 설명 |
|----------|------|----------|
| **Episode Termination** | 승리 또는 패배 | **승리**: 모든 타일 제거<br>**패배**: 시간 소진 또는 유효 이동 없음 |
| **Max Steps per Episode** | 200 | 무한 루프 방지 |
| **Initial State** | 랜덤 보드 생성 | BoardGenerator 사용<br>10색상 × 20타일 = 200개 |
| **Time Limit** | 120초 | 게임 규칙과 동일 |

**Max Steps 설정 근거:**
- 총 타일 수(200개)와 동일하게 설정하여 충분한 시도 기회 제공
- 이론적 최소 step 수는 100회(매번 2개씩 제거)이지만, 학습 초기 무효 action을 고려
- 학습 후기에는 60-80 step 내에 완료 가능하지만, 초기 학습 단계의 exploration을 위한 여유 확보
- 실제로는 Time Limit(120초)가 먼저 도달하는 경우가 많을 것으로 예상

### 3.2 State

**State 정의:** Agent가 관찰하는 Environment의 현재 상황 (snapshot)

```python
# 옵션 1: 2D 그리드 인코딩 (추천)
state_space = spaces.Box(
    low=0,
    high=10,  # 0: 빈 셀, 1-10: 각 색상
    shape=(15, 23),  # rows x cols
    dtype=np.int8
)

# 옵션 2: One-hot 인코딩 (더 많은 정보)
state_space = spaces.Box(
    low=0,
    high=1,
    shape=(15, 23, 11),  # rows x cols x (10 colors + empty)
    dtype=np.float32
)
```

**추가 State 정보:**
- 남은 시간 (정규화된 0~1 값)
- 남은 타일 수 (정규화된 0~1 값)

### 3.3 Action

**Action 정의:** Agent가 Environment에서 수행하는 행동 (셀 선택)

```python
# 모든 셀 위치를 행동으로 표현
action_space = spaces.Discrete(345)  # 23 * 15 = 345

# 행동 -> 위치 변환
def action_to_position(action: int) -> Position:
    row = action // 23
    col = action % 23
    return Position(row, col)
```

### 3.4 Reward

**Reward 정의:** Agent의 Action에 대해 Environment가 제공하는 피드백 (점수)

#### 3.4.1 Reward 구성 요소

| Reward 유형 | 조건 | Reward 값 | 목적 |
|----------|------|--------|------|
| **타일 제거 Reward** | 유효한 action으로 타일 제거 | `+1.0 × 타일 수` | 타일 제거 행동 강화 |
| **무효 이동 Penalty** | (1) 타일이 있는 셀 클릭<br>(2) 제거 가능한 타일이 없는 경우 | `-10.0` | 잘못된 action 억제 |
| **승리 Bonus** | 모든 타일 제거 성공 | `+100.0` | 목표 달성 강화 |
| **패배 Penalty** | 시간 소진 또는 막힘 | `-(남은 타일 × 10)` | 많은 타일 제거 유도 |

#### 3.4.2 Reward 계산 예시

| 상황 | 계산 | 총 보상 |
|------|------|---------|
| 2개 타일 제거 | 2×1.0 = 2.0 | **+2.0** |
| 3개 타일 제거 | 3×1.0 = 3.0 | **+3.0** |
| 4개 타일 제거 | 4×1.0 = 4.0 | **+4.0** |
| 무효 이동 | -10.0 | **-10.0** |
| 게임 승리 (200개 제거) | 200×1.0 + 100.0 = 300.0 | **+300.0** |
| 게임 패배 (50개 남음) | 150×1.0 - (50×10) = -350.0 | **-350.0** |

#### 3.4.3 Reward 계산 코드

```python
def calculate_reward(move_result: MoveResult, game_state: GameState,
                     remaining_tiles: int) -> float:
    """
    Agent의 action에 대한 reward를 계산

    Args:
        move_result: Action 실행 결과
        game_state: 현재 게임 상태
        remaining_tiles: 남은 타일 수

    Returns:
        reward: Agent에게 제공할 피드백 값
    """
    reward = 0.0

    # 1. 타일 제거 reward
    if move_result.success:
        tiles_removed = len(move_result.removed_tiles)
        reward += tiles_removed * 1.0  # 제거된 타일 수에 비례
    else:
        # 2. 무효 action penalty
        reward -= 10.0

    # 3. Episode 종료 reward
    if game_state == GameState.WON:
        reward += 100.0  # 승리 bonus
    elif game_state in [GameState.LOST_TIME, GameState.LOST_NO_MOVES]:
        reward -= remaining_tiles * 10  # 남은 타일에 비례한 패배 penalty

    return reward
```

---

## 4. 신경망 아키텍처

### 4.1 CNN 기반 Policy Network

```python
class ColorTilesPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 15 * 23 + 3, 256),  # +3 for time, tiles, valid_moves
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor (policy) head
        self.actor = nn.Linear(128, 345)  # 345 actions

        # Critic (value) head
        self.critic = nn.Linear(128, 1)
```

### 4.2 모델 크기 예상
- 총 파라미터: 약 15M
- 체크포인트 크기: 약 60MB

---

## 5. Checkpoint 시스템

### 5.1 Checkpoint 저장 구조

```
checkpoints/
├── ppo_colortiles_step_10000.zip
├── ppo_colortiles_step_50000.zip
├── ppo_colortiles_step_100000.zip
├── ppo_colortiles_best.zip          # 최고 성능 model
├── ppo_colortiles_latest.zip        # 최신 model
└── training_log.csv                  # Training 로그
```

### 5.2 Checkpoint 저장 조건

1. **Step 기반**: 매 10,000 step마다 자동 저장
2. **성능 기반**: 평균 reward가 이전 최고치를 갱신할 때
3. **Episode 기반**: 매 100 episode마다
4. **수동 저장**: GUI에서 버튼으로 저장

### 5.3 Checkpoint Metadata

```python
checkpoint_metadata = {
    "step": 50000,
    "episode": 1250,
    "mean_reward": 45.6,
    "win_rate": 0.35,
    "avg_tiles_cleared": 156.2,
    "avg_episode_length": 89.3,
    "timestamp": "2024-01-15T10:30:00",
    "hyperparameters": {...}
}
```

---

## 6. GUI 통합 계획

### 6.1 AI 플레이어 모드 UI

```
┌──────────────────────────────────────────────────────────────┐
│  [보드 위젯 23x15]              │  Color Tiles               │
│                                 │  ─────────────────         │
│                                 │  점수: 45                  │
│                                 │  시간: 85초                │
│                                 │  타일: 156개               │
│                                 │  상태: AI 플레이 중        │
│                                 │  ─────────────────         │
│                                 │  [게임 시작]               │
│                                 │  [새 게임]                 │
│                                 │  ─────────────────         │
│                                 │  AI 설정                   │
│                                 │  [체크포인트 선택 ▼]       │
│                                 │  [AI 플레이] [중지]        │
│                                 │  속도: [느림|보통|빠름]    │
│                                 │  ─────────────────         │
│                                 │  AI 상태                   │
│                                 │  스텝: 45                  │
│                                 │  예상 가치: 0.75           │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 새로운 GUI 컴포넌트

#### 6.2.1 AIControlPanel

```python
class AIControlPanel(QWidget):
    """AI 플레이어 제어 패널"""

    def __init__(self):
        # 체크포인트 선택 드롭다운
        self.checkpoint_combo = QComboBox()

        # AI 플레이 시작/중지 버튼
        self.play_button = QPushButton("AI 플레이")
        self.stop_button = QPushButton("중지")

        # 플레이 속도 조절
        self.speed_slider = QSlider()

        # 다음 행동 보기 (디버그용)
        self.show_next_action = QCheckBox("다음 행동 표시")
```

#### 6.2.2 AIStatusPanel

```python
class AIStatusPanel(QWidget):
    """AI 상태 표시 패널"""

    def update_status(self, ai_info: dict):
        self.step_label.setText(f"스텝: {ai_info['step']}")
        self.value_label.setText(f"예상 가치: {ai_info['value']:.2f}")
        self.action_probs_view.update(ai_info['action_probs'])
```

#### 6.2.3 다음 행동 시각화

```python
def highlight_next_action(self, position: Position, confidence: float):
    """AI가 선택할 다음 위치를 보드에 하이라이트"""
    button = self.cell_buttons[position.row][position.col]
    # 신뢰도에 따라 하이라이트 강도 조절
    highlight_color = self._get_highlight_color(confidence)
    button.setStyleSheet(f"border: 3px solid {highlight_color};")
```

### 6.3 Checkpoint 로딩 및 실행

```python
class AIPlayer:
    """학습된 Agent를 로드하여 게임 플레이"""

    def __init__(self, checkpoint_path: str):
        self.model = PPO.load(checkpoint_path)  # 학습된 Agent 로드

    def get_action(self, state: np.ndarray) -> tuple[int, float, np.ndarray]:
        """
        현재 State에서 Agent가 선택할 Action 결정

        Args:
            state: 현재 Environment의 state

        Returns:
            action: Agent가 선택한 action (셀 인덱스 0-344)
            value: State value 추정값
            action_probs: 각 action의 확률 분포
        """
        # Agent가 state를 보고 action 예측
        action, _states = self.model.predict(
            state,
            deterministic=True  # 평가 모드: 가장 높은 확률의 action 선택
        )

        # 추가 정보 추출 (GUI 시각화용)
        value = self.model.policy.predict_values(state)
        action_probs = self.model.policy.get_distribution(state).distribution.probs

        return action, value, action_probs
```

---

## 7. 성능 측정 지표

### 7.1 Primary Metrics (주요 지표)

| Metric | 설명 | 목표값 |
|------|------|--------|
| **Win Rate** | Episode 승리 비율 | > 80% |
| **Average Score** | Episode당 평균 획득 점수 | > 180 |
| **Average Tiles Cleared** | Episode당 제거한 타일 수 | > 190 |
| **Average Game Time** | 승리 시 소요 시간 | < 60초 |

### 7.2 Training Metrics (학습 지표)

| Metric | 설명 | 용도 |
|------|------|------|
| **Episode Reward** | Episode 총 reward | 학습 진행도 |
| **Policy Loss** | Policy network 손실 | 학습 안정성 |
| **Value Loss** | Value network 손실 | 가치 추정 정확도 |
| **Entropy** | Policy entropy | Exploration 정도 |
| **KL Divergence** | Policy 변화량 | 업데이트 안정성 |
| **Clip Fraction** | PPO clipping 비율 | Learning rate 적절성 |

### 7.3 Efficiency Metrics (효율성 지표)

| Metric | 설명 |
|------|------|
| **Invalid Move Rate** | 전체 action 중 무효 이동 비율 (낮을수록 좋음) |
| **Actions per Tile** | 타일 하나 제거에 필요한 평균 action 수 |
| **Multi-Remove Rate** | 3개 이상 타일을 한번에 제거한 비율 |

### 7.4 Performance Tracking 구현

```python
class PerformanceTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record_episode(self, episode_info: dict):
        """Episode 종료 시 metric 기록"""
        self.metrics['win'].append(episode_info['won'])
        self.metrics['score'].append(episode_info['score'])
        self.metrics['tiles_cleared'].append(episode_info['tiles_cleared'])
        self.metrics['invalid_moves'].append(episode_info['invalid_moves'])
        self.metrics['episode_length'].append(episode_info['steps'])
        self.metrics['game_time'].append(episode_info['game_time'])

    def get_summary(self, last_n: int = 100) -> dict:
        """최근 n개 episode의 요약 통계"""
        return {
            'win_rate': np.mean(self.metrics['win'][-last_n:]),
            'avg_score': np.mean(self.metrics['score'][-last_n:]),
            'avg_tiles_cleared': np.mean(self.metrics['tiles_cleared'][-last_n:]),
            'invalid_move_rate': np.mean(self.metrics['invalid_moves'][-last_n:]) /
                                np.mean(self.metrics['episode_length'][-last_n:]),
            'avg_game_time': np.mean([t for t, w in
                zip(self.metrics['game_time'][-last_n:],
                    self.metrics['win'][-last_n:]) if w])
        }
```

### 7.5 TensorBoard 시각화

```python
# TensorBoard 로깅
tensorboard_callback = TensorBoardCallback(log_dir="./logs/")

# 커스텀 콜백으로 추가 메트릭 로깅
class MetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            summary = self.performance_tracker.get_summary()
            for key, value in summary.items():
                self.logger.record(f"custom/{key}", value)
        return True
```

---

## 8. 구현 단계

### Phase 1: 환경 구현
- [ ] Gymnasium 환경 래퍼 (`ColorTilesEnv`) 구현
- [ ] State 인코딩 함수 구현
- [ ] Reward 함수 구현 (무효 action penalty 포함)
- [ ] Environment 단위 테스트

### Phase 2: 학습 파이프라인
- [ ] PPO 모델 설정
- [ ] 체크포인트 저장/로드 시스템
- [ ] 성능 추적 시스템
- [ ] TensorBoard 로깅
- [ ] 학습 스크립트 작성

### Phase 3: GUI 통합
- [ ] AIControlPanel 구현
- [ ] 체크포인트 선택 UI
- [ ] AI 플레이어 클래스
- [ ] 실시간 행동 시각화
- [ ] 속도 조절 기능

### Phase 4: 튜닝 및 최적화
- [ ] 하이퍼파라미터 튜닝
- [ ] 보상 함수 조정
- [ ] 네트워크 아키텍처 실험
- [ ] 최종 모델 학습

---

## 9. Hyperparameters 초기 설정

```python
ppo_config = {
    # Training 관련
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE parameter

    # PPO 특화
    "clip_range": 0.2,  # PPO clip epsilon
    "clip_range_vf": None,
    "ent_coef": 0.01,  # Entropy coefficient (exploration)
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Gradient clipping

    # Training Environment
    "n_envs": 8,  # 병렬 environment 수
    "total_timesteps": 1_000_000,

    # Checkpoint
    "save_freq": 10000,
    "eval_freq": 5000,
    "eval_episodes": 20,
}
```

---

## 10. 기대 결과

### 10.1 Learning Curve 예상

| 학습 단계 | Training Steps | 예상 Win Rate | 평균 Tiles Cleared | Invalid Move Rate |
|------|---------|-----------|----------------|-------------------|
| 초기 (Early) | 0-50K | 0-5% | 30-50 | **60-80%** |
| 초중기 (Early-Mid) | 50K-200K | 5-30% | 80-120 | **30-50%** |
| 중기 (Mid) | 200K-500K | 30-60% | 140-170 | **15-25%** |
| 후기 (Late) | 500K-1M | 60-80% | 175-195 | **5-10%** |
| 수렴 (Converged) | 1M+ | 80%+ | 195+ | **< 5%** |

**학습 진행 특징:**
- 초기에는 무효 action을 자주 시도하며 penalty를 받음 (60-80%)
- 학습이 진행되면서 Invalid Move Rate가 급격히 감소
- Agent가 state를 이해하고 유효한 action을 구분하는 법을 학습
- 최종적으로 5% 미만의 무효 이동률 달성 예상

### 10.2 최종 목표
- **Win Rate**: 85% 이상
- **Average Score**: 190점 이상
- **Invalid Move Rate**: 5% 미만
- **Average Win Time**: 50초 이내

---

## 11. 필요 라이브러리

```txt
# requirements-rl.txt
gymnasium>=0.29.0
stable-baselines3>=2.2.0
torch>=2.0.0
tensorboard>=2.14.0
numpy>=1.24.0
```

---

## 12. 디렉토리 구조 (최종)

```
color-tiles-rl/
├── src/
│   ├── color_tiles/
│   │   ├── domain/
│   │   ├── engine/
│   │   ├── gui/
│   │   │   ├── ai_control_panel.py    # NEW
│   │   │   ├── ai_status_panel.py     # NEW
│   │   │   └── ...
│   │   └── utils/
│   └── rl/                             # NEW
│       ├── env/
│       │   └── color_tiles_env.py     # Gymnasium 환경
│       ├── models/
│       │   └── policy.py              # 커스텀 정책 네트워크
│       ├── training/
│       │   ├── train.py               # 학습 스크립트
│       │   └── callbacks.py           # 커스텀 콜백
│       └── inference/
│           └── ai_player.py           # AI 플레이어
├── checkpoints/                        # NEW
├── logs/                               # NEW (TensorBoard)
├── docs/
│   └── reinforce_learning_plan.md
└── main.py
```
