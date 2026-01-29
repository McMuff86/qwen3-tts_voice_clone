"""
Qwen3-TTS Voice Cloning - Automatisches Setup Script
=====================================================
Erstellt Conda/venv Environment, installiert Dependencies und
laedt Modelle von HuggingFace herunter.

Usage: python setup.py
"""

import os
import sys
import shutil
import subprocess
import platform


# ── Konstanten ──────────────────────────────────────────────────────────────

CONDA_ENV_NAME = "qwen3-tts"
PYTHON_VERSION = "3.12"
VENV_DIR = ".venv"
REQUIREMENTS_FILE = "requirements.txt"

MODELS = [
    {
        "name": "Qwen3-TTS-Tokenizer-12Hz",
        "repo": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "size": "~650 MB",
        "desc": "Tokenizer (benoetigt von allen Modellen)",
    },
    {
        "name": "Qwen3-TTS-12Hz-1.7B-Base",
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "size": "~4.3 GB",
        "desc": "Voice Cloning (1.7B)",
    },
    {
        "name": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "size": "~4.3 GB",
        "desc": "Custom Voices (1.7B)",
    },
    {
        "name": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "size": "~4.3 GB",
        "desc": "Voice Design (1.7B)",
    },
    {
        "name": "Qwen3-TTS-12Hz-0.6B-Base",
        "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "size": "~2.5 GB",
        "desc": "Voice Cloning (0.6B, kleiner)",
    },
    {
        "name": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "size": "~2.5 GB",
        "desc": "Custom Voices (0.6B, kleiner)",
    },
]


# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def print_header(text: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_step(text: str) -> None:
    print(f"\n>> {text}")


def print_ok(text: str) -> None:
    print(f"   [OK] {text}")


def print_warn(text: str) -> None:
    print(f"   [!!] {text}")


def print_fail(text: str) -> None:
    print(f"   [FEHLER] {text}")


def has_command(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f"   $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [J/n] " if default else " [j/N] "
    while True:
        answer = input(prompt + suffix).strip().lower()
        if answer == "":
            return default
        if answer in ("j", "ja", "y", "yes"):
            return True
        if answer in ("n", "nein", "no"):
            return False
        print("   Bitte 'j' oder 'n' eingeben.")


# ── 1. Python-Version pruefen ──────────────────────────────────────────────

def check_python_version() -> None:
    print_step("Python-Version pruefen...")
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"   Python {major}.{minor}.{sys.version_info.micro}")

    if major != 3 or minor < 10 or minor > 12:
        print_fail(f"Python 3.10-3.12 wird benoetigt (gefunden: {major}.{minor})")
        print("   Bitte installiere eine kompatible Python-Version.")
        sys.exit(1)

    print_ok("Python-Version kompatibel")


# ── 2. Verfuegbare Tools erkennen ──────────────────────────────────────────

def detect_tools() -> dict[str, bool]:
    print_step("Verfuegbare Tools erkennen...")
    tools = {
        "conda": has_command("conda"),
        "uv": has_command("uv"),
        "pip": has_command("pip"),
        "huggingface-cli": has_command("huggingface-cli"),
    }
    for name, available in tools.items():
        status = "gefunden" if available else "nicht gefunden"
        (print_ok if available else print_warn)(f"{name}: {status}")
    return tools


# ── 3. Environment erstellen ───────────────────────────────────────────────

def setup_environment(tools: dict[str, bool]) -> None:
    print_header("Environment einrichten")

    if tools["conda"]:
        # Pruefen ob Conda-Env bereits existiert
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, check=False,
        )
        env_exists = any(
            CONDA_ENV_NAME == line.split()[0]
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("#")
        )

        if env_exists:
            print_ok(f"Conda-Environment '{CONDA_ENV_NAME}' existiert bereits.")
            return

        if ask_yes_no(f"   Conda-Environment '{CONDA_ENV_NAME}' erstellen?"):
            print_step(f"Erstelle Conda-Environment '{CONDA_ENV_NAME}'...")
            run(["conda", "create", "-n", CONDA_ENV_NAME,
                 f"python={PYTHON_VERSION}", "-y"])
            print_ok(f"Environment '{CONDA_ENV_NAME}' erstellt.")
            print()
            print("   Aktiviere das Environment mit:")
            print(f"     conda activate {CONDA_ENV_NAME}")
            print()
            print("   Dann fuehre dieses Script erneut aus:")
            print("     python setup.py")
            print()

            if not ask_yes_no("   Trotzdem mit aktuellem Python fortfahren?", default=False):
                sys.exit(0)
        return

    # Fallback: venv
    venv_path = os.path.join(os.getcwd(), VENV_DIR)
    if os.path.isdir(venv_path):
        print_ok(f"venv '{VENV_DIR}' existiert bereits.")
        return

    if ask_yes_no(f"   Virtuelles Environment '{VENV_DIR}' erstellen?"):
        print_step("Erstelle venv...")
        if tools["uv"]:
            run(["uv", "venv", VENV_DIR])
        else:
            run([sys.executable, "-m", "venv", VENV_DIR])

        print_ok(f"Environment '{VENV_DIR}' erstellt.")
        print()

        if platform.system() == "Windows":
            activate_cmd = f"  {VENV_DIR}\\Scripts\\activate"
        else:
            activate_cmd = f"  source {VENV_DIR}/bin/activate"

        print("   Aktiviere das Environment mit:")
        print(f"   {activate_cmd}")
        print()
        print("   Dann fuehre dieses Script erneut aus:")
        print("     python setup.py")
        print()

        if not ask_yes_no("   Trotzdem mit aktuellem Python fortfahren?", default=False):
            sys.exit(0)


# ── 4. Dependencies installieren ───────────────────────────────────────────

def install_dependencies(tools: dict[str, bool]) -> None:
    print_header("Dependencies installieren")

    req_path = os.path.join(os.getcwd(), REQUIREMENTS_FILE)
    if not os.path.isfile(req_path):
        print_fail(f"'{REQUIREMENTS_FILE}' nicht gefunden!")
        return

    if not ask_yes_no(f"   Dependencies aus '{REQUIREMENTS_FILE}' installieren?"):
        print("   Uebersprungen.")
        return

    print_step("Installiere Dependencies...")
    if tools["uv"]:
        run(["uv", "pip", "install", "-r", REQUIREMENTS_FILE])
    elif tools["pip"]:
        run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    else:
        print_fail("Weder uv noch pip gefunden! Kann Dependencies nicht installieren.")
        return

    print_ok("Dependencies installiert.")

    # Flash Attention optional anbieten
    print()
    if ask_yes_no("   Flash Attention installieren? (benoetigt CUDA 12.4+)", default=False):
        print_step("Installiere Flash Attention...")
        install_cmd = (
            ["uv", "pip", "install", "flash-attn"]
            if tools["uv"]
            else [sys.executable, "-m", "pip", "install", "flash-attn"]
        )
        result = run(install_cmd, check=False)
        if result.returncode == 0:
            print_ok("Flash Attention installiert.")
        else:
            print_warn("Flash Attention konnte nicht installiert werden.")
            print("   Siehe: https://huggingface.co/lldacing/flash-attention-windows-wheel")


# ── 5. Modelle herunterladen ───────────────────────────────────────────────

def download_models(tools: dict[str, bool]) -> None:
    print_header("Modelle herunterladen")

    # Pruefen ob huggingface-cli verfuegbar ist
    if not tools["huggingface-cli"]:
        print_warn("huggingface-cli nicht gefunden.")
        print("   Installiere es mit: pip install huggingface_hub[cli]")
        if ask_yes_no("   Jetzt installieren?"):
            install_cmd = (
                ["uv", "pip", "install", "huggingface_hub[cli]"]
                if tools["uv"]
                else [sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"]
            )
            result = run(install_cmd, check=False)
            if result.returncode != 0:
                print_fail("Installation fehlgeschlagen. Modell-Download uebersprungen.")
                return
            tools["huggingface-cli"] = True
        else:
            print("   Modell-Download uebersprungen.")
            return

    # Bereits heruntergeladene Modelle erkennen
    print()
    print("   Verfuegbare Modelle:")
    print("   " + "-" * 56)
    for i, model in enumerate(MODELS, 1):
        local_dir = os.path.join(os.getcwd(), model["name"])
        exists = os.path.isdir(local_dir)
        marker = " [vorhanden]" if exists else ""
        print(f"   {i}) {model['name']}")
        print(f"      {model['desc']} ({model['size']}){marker}")
    print("   " + "-" * 56)
    print()
    print("   Eingabe: Nummern mit Komma getrennt (z.B. 1,2,3)")
    print("            'all' fuer alle, 'none' zum Ueberspringen")
    print()

    while True:
        choice = input("   Auswahl: ").strip().lower()
        if choice in ("none", "n", ""):
            print("   Modell-Download uebersprungen.")
            return
        if choice == "all":
            selected = list(range(len(MODELS)))
            break

        try:
            selected = [int(x.strip()) - 1 for x in choice.split(",")]
            if all(0 <= s < len(MODELS) for s in selected):
                break
            print(f"   Bitte Nummern zwischen 1 und {len(MODELS)} eingeben.")
        except ValueError:
            print("   Ungueltige Eingabe. Bitte Nummern oder 'all'/'none' eingeben.")

    print()
    for idx in selected:
        model = MODELS[idx]
        local_dir = os.path.join(os.getcwd(), model["name"])

        if os.path.isdir(local_dir):
            print_ok(f"{model['name']} bereits vorhanden, ueberspringe.")
            continue

        print_step(f"Lade {model['name']} ({model['size']})...")
        result = run(
            ["huggingface-cli", "download", model["repo"],
             "--local-dir", local_dir],
            check=False,
        )
        if result.returncode == 0:
            print_ok(f"{model['name']} heruntergeladen.")
        else:
            print_fail(f"{model['name']} konnte nicht heruntergeladen werden.")

    print_ok("Modell-Download abgeschlossen.")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print_header("Qwen3-TTS Voice Cloning - Setup")

    # In Script-Verzeichnis wechseln
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    check_python_version()
    tools = detect_tools()
    setup_environment(tools)
    install_dependencies(tools)
    download_models(tools)

    print_header("Setup abgeschlossen!")
    print()
    print("   Naechste Schritte:")
    print("   1. Environment aktivieren (falls noch nicht geschehen)")
    print("   2. Jupyter Notebook oder Gradio UI starten")
    print()


if __name__ == "__main__":
    main()
