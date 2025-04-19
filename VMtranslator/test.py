
from pathlib import Path
import subprocess

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Test VM Translator")
    parser.add_argument("path")
    args = parser.parse_args()
    user_path = Path(args.path)

    for file in user_path.rglob("*.vm"):
        output_path = Path("out_asm") / file.parent / f"{file.stem}.asm"
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

        print(f"Testing VM translator on {file}")
        ret = subprocess.run(
            ["python3", "vm_translator.py", str(file), "-o", str(output_path)],
            capture_output=True
        )
        print(f"OUT: {ret.stdout.decode().strip()}")
        print(f"ERR:\033[31m{ret.stderr.decode().strip()}\033[0m")
        print("________________________________________________________\n")
