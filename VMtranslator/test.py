from pathlib import Path
import subprocess

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Test VM Translator")
    parser.add_argument("path")
    args = parser.parse_args()
    user_path = Path(args.path)

    for file in user_path.rglob("*.vm"):
        print(f"Testing VM translator on {file}")
        ret = subprocess.run(["python3", "vm_translator.py", file, "-o", f"out_asm/{file.stem}.asm"],capture_output=True)
        print(f"OUT: {str(ret.stdout)}")
        print(f"ERR:\033[31m{str(ret.stderr)}\033[0m")
        print("________________________________________________________\n")
