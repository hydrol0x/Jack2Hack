from assembler import assemble


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="hasm",
        description="Assembles Hack ASM code to the Hack machine code",
    )
    parser.add_argument(
        "filepath", type=str, help="Filepath to input Hack assembly file"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.hack",
        type=str,
        help="Output file that the machine code will be written to",
    )

    args = parser.parse_args()

    # try:
    machine_program = assemble(args.filepath)
    with open(args.output, "w") as file:
        # Write each element of the list to the file followed by a newline character
        for item in machine_program:
            file.write(item + "\n")
    # except:
    # print("Assembly failed.")
else:
    print("Library found at assembler.py")
    print("usage: hasm [-h] [-o OUTPUT] filepath")
