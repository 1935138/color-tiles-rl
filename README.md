# Color Tiles Game Engine

순수 Python 게임 로직 엔진으로, GUI(PyQt6)와 완전히 분리된 Color Tiles 퍼즐 게임 구현입니다.

## 게임 규칙

- **보드**: 23 × 15 그리드 (총 345칸)
- **타일**: 10가지 색상 × 20개씩 = 200개
- **빈칸**: 145개
- **제한시간**: 120초
- **조작**: 빈칸을 클릭하여 타일 제거
- **제거 조건**: 클릭한 빈칸 기준 상하좌우 4방향에서 찾은 타일 중 같은 색상이 2개 이상이면 제거
- **점수**: 타일 1개당 1점
- **패널티**: 잘못된 이동 시 시간 10초 감소
- **승리**: 모든 타일 제거
- **패배**: 시간 초과 또는 유효한 이동 없음

## 설치

### 요구사항

- Python 3.10 이상
- uv (빠른 Python 패키지 매니저)

### uv 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip로 설치
pip install uv
```

### 프로젝트 설정

```bash
# 프로젝트 클론
git clone <repository-url>
cd color_tiles

# uv로 가상환경 생성
uv venv

# 가상환경 활성화
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 의존성 설치 (PyQt6)
uv pip install -e .
```

## 빠른 시작

### CLI 데모 실행

```bash
# uv로 실행 (가상환경 자동 활성화)
uv run python examples/cli_demo.py

# 또는 가상환경 활성화 후
source .venv/bin/activate
python examples/cli_demo.py
```

터미널에서 간단한 텍스트 기반 게임을 플레이할 수 있습니다.

### 기본 사용법

```python
from color_tiles.utils.board_generator import BoardGenerator
from color_tiles.engine.game import GameEngine
from color_tiles.domain.models import Position

# 1. 랜덤 보드 생성
board = BoardGenerator.generate_random_board()

# 2. 게임 엔진 초기화
game = GameEngine(board)

# 3. 게임 시작
game.start_game()

# 4. 이동 실행
position = Position(row=5, col=10)
result = game.make_move(position)

print(f"성공: {result.success}")
print(f"제거된 타일: {len(result.tiles_removed)}")
print(f"획득 점수: {result.points_earned}")
print(f"메시지: {result.message}")

# 5. 게임 상태 조회
print(f"현재 점수: {game.get_score()}")
print(f"남은 시간: {game.get_remaining_time():.1f}초")
print(f"게임 상태: {game.get_game_state()}")
```

## 아키텍처

### Clean Architecture

프로젝트는 3개 계층으로 구성됩니다:

```
src/color_tiles/
├── domain/          # 핵심 데이터 모델 및 상수
│   ├── models.py    # Color, Position, Cell, GameState, MoveResult
│   ├── constants.py # 게임 상수
│   └── exceptions.py# 커스텀 예외
├── engine/          # 게임 로직 및 상태 관리
│   ├── board.py     # Board 상태 관리
│   ├── tile_finder.py    # 4방향 타일 찾기 알고리즘
│   ├── move_validator.py # 이동 유효성 검증
│   └── game.py      # GameEngine (메인 Public API)
└── utils/           # 유틸리티
    └── board_generator.py # 랜덤 보드 생성
```

## API 문서

### 핵심 클래스

#### GameEngine

메인 게임 오케스트레이터로, GUI가 사용할 Primary API입니다.

```python
class GameEngine:
    def __init__(self, board: Board, time_limit: float = 120.0)

    # 게임 생명주기
    def start_game(self) -> None
    def reset_game(self, new_board: Optional[Board] = None) -> None

    # 게임 진행
    def make_move(self, position: Position) -> MoveResult

    # 상태 조회
    def get_game_state(self) -> GameState
    def get_remaining_time(self) -> float
    def get_score(self) -> int
    def get_board_state(self) -> dict
    def get_board(self) -> Board

    # 옵저버 패턴
    def add_observer(self, observer: GameObserver) -> None
    def remove_observer(self, observer: GameObserver) -> None
```

#### Board

보드 상태를 관리하는 핵심 클래스입니다.

```python
class Board:
    def __init__(self, cells: list[list[Optional[Color]]])

    def get_cell(self, position: Position) -> Cell
    def is_empty(self, position: Position) -> bool
    def remove_tiles(self, positions: list[Position]) -> int
    def get_all_tiles(self) -> list[Cell]
    def count_tiles(self) -> int
    def to_dict(self) -> dict
    def copy(self) -> Board
```

#### BoardGenerator

랜덤 게임 보드를 생성합니다.

```python
class BoardGenerator:
    @staticmethod
    def generate_random_board() -> Board
```

### 데이터 모델

#### Position

```python
@dataclass(frozen=True)
class Position:
    row: int
    col: int
```

#### Cell

```python
@dataclass(frozen=True)
class Cell:
    position: Position
    color: Optional[Color]  # None = 빈칸

    @property
    def is_empty(self) -> bool
```

#### GameState

```python
class GameState(Enum):
    READY = "ready"
    PLAYING = "playing"
    WON = "won"
    LOST_TIME = "lost_time"
    LOST_NO_MOVES = "lost_no_moves"
```

#### MoveResult

```python
@dataclass
class MoveResult:
    success: bool
    tiles_removed: list[Cell]
    points_earned: int
    time_penalty: float
    message: str
    game_state: GameState
```

## PyQt6 GUI 통합 예제

```python
from PyQt6.QtCore import QTimer
from color_tiles.utils.board_generator import BoardGenerator
from color_tiles.engine.game import GameEngine, GameObserver
from color_tiles.domain.models import Position, GameState

class PyQt6Observer(GameObserver):
    """PyQt6 GUI를 위한 옵저버."""

    def __init__(self, ui):
        self.ui = ui

    def on_move_made(self, result):
        """이동 완료 시 UI 업데이트."""
        self.ui.update_board()
        self.ui.update_score(self.ui.game.get_score())

        if not result.success:
            self.ui.show_message(result.message)

    def on_game_state_changed(self, state):
        """게임 상태 변경 시 UI 업데이트."""
        if state == GameState.WON:
            self.ui.show_victory_dialog()
        elif state in [GameState.LOST_TIME, GameState.LOST_NO_MOVES]:
            self.ui.show_game_over_dialog()

    def on_time_updated(self, remaining):
        """시간 업데이트 (QTimer에서 주기적으로 호출)."""
        self.ui.update_timer_display(remaining)

class ColorTilesUI:
    """PyQt6 GUI 예제."""

    def __init__(self):
        # 게임 초기화
        board = BoardGenerator.generate_random_board()
        self.game = GameEngine(board)

        # 옵저버 등록
        self.observer = PyQt6Observer(self)
        self.game.add_observer(self.observer)

        # 타이머 설정 (100ms마다 시간 업데이트)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(100)

        # 게임 시작
        self.game.start_game()

    def update_time(self):
        """주기적으로 남은 시간을 확인하고 UI 업데이트."""
        remaining = self.game.get_remaining_time()
        self.observer.on_time_updated(remaining)

    def on_cell_clicked(self, row, col):
        """셀 클릭 핸들러."""
        position = Position(row, col)
        result = self.game.make_move(position)
        # 옵저버가 자동으로 UI 업데이트 수행

    def update_board(self):
        """보드 상태를 읽어 UI 업데이트."""
        board_state = self.game.get_board_state()
        # board_state['cells'] 사용하여 GUI 그리기

    def update_score(self, score):
        """점수 표시 업데이트."""
        pass

    def update_timer_display(self, remaining):
        """타이머 표시 업데이트."""
        pass

    def show_victory_dialog(self):
        """승리 다이얼로그 표시."""
        pass

    def show_game_over_dialog(self):
        """게임 오버 다이얼로그 표시."""
        pass

    def show_message(self, message):
        """메시지 표시."""
        pass
```

## Observer Pattern

게임 엔진은 Observer Pattern을 사용하여 GUI에 이벤트를 알립니다.

```python
from color_tiles.engine.game import GameObserver

class CustomObserver(GameObserver):
    def on_move_made(self, result):
        """이동 완료 후 호출."""
        print(f"Move: {result.message}")

    def on_game_state_changed(self, state):
        """게임 상태 변경 시 호출."""
        print(f"State changed to: {state.value}")

    def on_time_updated(self, remaining):
        """시간 업데이트 시 호출."""
        print(f"Time remaining: {remaining:.1f}s")

# 옵저버 등록
observer = CustomObserver()
game.add_observer(observer)
```

## 핵심 알고리즘

### 타일 찾기 (4방향 탐색)

빈칸에서 상/하/좌/우 4방향으로 탐색하여 각 방향의 첫 번째 타일을 찾습니다.

```python
# src/color_tiles/engine/tile_finder.py:find_tiles_from_position()
# 시간 복잡도: O(max(width, height)) × 4 = O(23) × 4 = O(92)
```

### 이동 검증

찾은 타일들을 색상별로 그룹화하여 2개 이상인 색상이 있는지 확인합니다.

```python
# src/color_tiles/engine/move_validator.py:is_valid_move()
# 시간 복잡도: O(4) for grouping
```

### 유효한 이동 탐색

모든 빈칸을 순회하며 유효한 이동이 있는지 확인합니다.

```python
# src/color_tiles/engine/move_validator.py:find_all_valid_moves()
# 시간 복잡도: O(width × height × max(width, height))
#             = O(345 × 23) ≈ O(8,000)
```

## 성능 특성

- **보드 셀 접근**: O(1)
- **타일 찾기**: O(max(width, height)) = O(23)
- **이동 검증**: O(4) for grouping
- **전체 유효 이동 탐색**: O(width × height × max(width, height)) ≈ O(8,000)
- **메모리 사용**: ~수 KB (345개 셀)

실시간 게임에 충분한 성능을 제공합니다.

## 프로젝트 구조

```
color_tiles/
├── docs/
│   └── game_rule.md           # 게임 규칙 문서
├── src/
│   └── color_tiles/
│       ├── __init__.py
│       ├── domain/            # Domain Layer
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── models.py
│       │   └── exceptions.py
│       ├── engine/            # Engine Layer
│       │   ├── __init__.py
│       │   ├── board.py
│       │   ├── tile_finder.py
│       │   ├── move_validator.py
│       │   └── game.py
│       └── utils/             # Utils Layer
│           ├── __init__.py
│           └── board_generator.py
├── examples/
│   └── cli_demo.py            # CLI 데모
├── README.md
└── requirements.txt
```

## 🎮 PyQt6 GUI 실행하기

```bash
# 1. uv로 가상환경 및 의존성 설치 (처음 한 번만)
uv venv
uv pip install -e .

# 2. GUI 실행
uv run python main.py

# 또는 가상환경 활성화 후 실행
source .venv/bin/activate
python main.py
```

### 게임 플레이 방법
1. "게임 시작" 버튼 클릭
2. 빈칸(밝은 회색)을 클릭하여 타일 제거
3. 4방향에서 같은 색상 2개 이상이면 제거 성공
4. 모든 타일을 제거하면 승리!

## 향후 확장 가능성

현재 아키텍처는 다음 기능들을 지원할 수 있도록 설계되었습니다:

1. **Undo/Redo**: `Board.copy()`로 스냅샷 저장
2. **Save/Load**: `Board.to_dict()`로 직렬화
3. **Replay**: `MoveResult`에 모든 정보 포함
4. **AI Solver**: `find_all_valid_moves()`로 유효한 이동 탐색
5. **힌트 시스템**: `find_all_valid_moves()` 활용
6. **난이도 조절**: 시간 제한, 패널티 조정
7. **다른 보드 생성 전략**: `BoardGenerator` 교체

## 개발자 정보

- **버전**: 0.1.0
- **Python**: 3.10+
- **라이선스**: MIT (또는 프로젝트에 맞게 수정)

## 기여

이슈 리포트와 풀 리퀘스트를 환영합니다!

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. (또는 프로젝트에 맞게 수정)
