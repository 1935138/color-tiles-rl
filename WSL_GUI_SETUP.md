# WSL2에서 PyQt6 GUI 실행하기

## 문제 상황
WSL2에서 PyQt6 GUI를 실행하면 Qt 플랫폼 플러그인 오류가 발생합니다.

## 해결 방법

### 방법 1: WSLg 사용 (Windows 11 - 권장)

Windows 11에는 WSLg가 내장되어 있어 가장 간단합니다.

#### 1. 필요한 라이브러리 설치

```bash
sudo apt update
sudo apt install -y \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    libxkbcommon0
```

#### 2. WSL 재시작

Windows PowerShell에서:
```powershell
wsl --shutdown
```

그리고 WSL을 다시 시작합니다.

#### 3. 게임 실행

```bash
cd /home/jmlee/workspace/color_tiles
uv run python main.py
```

---

### 방법 2: VcXsrv (Windows 10 또는 Windows 11)

#### 1. Windows에 VcXsrv 설치

- https://sourceforge.net/projects/vcxsrv/ 에서 다운로드
- 설치 후 XLaunch 실행
- 설정:
  - Display number: 0
  - "Start no client" 선택
  - "Disable access control" 체크 (중요!)

#### 2. WSL에서 DISPLAY 환경 변수 설정

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1
```

적용:
```bash
source ~/.bashrc
```

#### 3. 필요한 라이브러리 설치 (방법 1과 동일)

```bash
sudo apt update
sudo apt install -y \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libxkbcommon-x11-0
```

#### 4. 게임 실행

```bash
cd /home/jmlee/workspace/color_tiles
uv run python main.py
```

---

### 방법 3: CLI 버전 사용 (GUI 없이)

GUI 설정이 번거롭다면 CLI 버전으로도 게임을 즐길 수 있습니다:

```bash
cd /home/jmlee/workspace/color_tiles
uv run python examples/cli_demo.py
```

---

## 트러블슈팅

### 오류: "qt.qpa.plugin: Could not load the Qt platform plugin"

**해결:**
```bash
# Qt 디버그 정보 출력
export QT_DEBUG_PLUGINS=1
uv run python main.py

# 특정 플랫폼 플러그인 강제 사용
export QT_QPA_PLATFORM=xcb
uv run python main.py
```

### 오류: "cannot connect to X server"

**해결:**
```bash
# DISPLAY 확인
echo $DISPLAY

# DISPLAY 설정 (VcXsrv 사용 시)
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# 또는 직접 설정
export DISPLAY=:0
```

### 오류: "Authorization required, but no authorization protocol specified"

**해결:**
```bash
# xhost 설치
sudo apt install x11-xserver-utils

# 접근 허용 (보안 주의!)
xhost +
```

---

## Windows 버전 확인

```powershell
# PowerShell에서
winver
```

- Windows 11: WSLg 사용 (방법 1)
- Windows 10: VcXsrv 사용 (방법 2)

---

## 권장 사항

1. **Windows 11 사용자**: WSLg (방법 1)를 사용하세요. 가장 간단하고 안정적입니다.
2. **Windows 10 사용자**: VcXsrv (방법 2)를 사용하세요.
3. **GUI 설정이 귀찮은 경우**: CLI 버전 (방법 3)을 사용하세요.
4. **네이티브 Windows**: Python과 PyQt6를 Windows에 직접 설치하여 실행하는 것도 좋은 방법입니다.
