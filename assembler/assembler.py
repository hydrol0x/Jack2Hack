"""
# Assembler for the Nand2Tetris assembly language
# Registers A, D
# @<value> or @<label> loads A register with <value> or the address that <lable> denotes 
# Can store data or address
#
# D register -> General data register
# M -> Data (word) in memory[A]. i.e the data stored at the memory address contained in A
"""


def remove_comments(program: list) -> list:
    for i, line in enumerate(program):
        newLine = line.split("//")[0]
        program[i] = newLine
    program = [line for line in program if line != ""]  # Removes all blank '' lines
    return program


def parse_file(path: str) -> list:
    program = []
    with open(path, "r") as file:
        for line in file:
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            if line:  # Doesn't add empty strings
                program.append(line)
            # print(line, end="")
    program = remove_comments(program)
    return program


def process_a_instruction(line: str) -> str:
    number = int(line.split("@")[1])
    bin_num = bin(number)[2:]  # Removes the 0b prefix
    return bin_num.zfill(16)


def process_c_instruction(line: str):
    # dest = comp ; jump
    # 111 a cccccc ddd jjj
    comp = ""
    dest = "null"
    jump = "null"
    a_bit = "0"
    comp_bits = "000000"
    dest_bits = "000"
    jump_bits = "000"
    if "=" in line:
        dest = line.split("=")[0]
        comp = line.split("=")[1]
    else:
        comp = line.split(";")[0]
    if ";" in line:
        jump = line.split(";")[1]

    # # Determine 'a' bit
    # if "M" in comp: # 'a' bit determines if M operations
    #     a_bit = "1"

    # Determine cccccc bits
    match comp:
        case "0":
            comp_bits = "101010"
        case "1":
            comp_bits = "111111"
        case "-1":
            comp_bits = "111010"
        case "D":
            comp_bits = "001100"
        case "A":
            comp_bits = "110000"
        case "M":
            comp_bits = "110000"
            a_bit = "1"
        case "!D":
            comp_bits = "001101"
        case "!A":
            comp_bits = "110001"
        case "!M":
            comp_bits = "110001"
            a_bit = "1"
        case "-D":
            comp_bits = "001111"
        case "-A":
            comp_bits = "110011"
        case "-M":
            comp_bits = "110011"
            a_bit = "1"
        case "D+1":
            comp_bits = "011111"
        case "A+1":
            comp_bits = "110111"
        case "M+1":
            comp_bits = "110111"
            a_bit = "1"
        case "D-1":
            comp_bits = "001110"
        case "A-1":
            comp_bits = "110010"
        case "M-1":
            comp_bits = "110010"
            a_bit = "1"
        case "D+A":
            comp_bits = "000010"
        case "D+M":
            comp_bits = "000010"
            a_bit = "1"
        case "D-A":
            comp_bits = "010011"
        case "D-M":
            comp_bits = "010011"
            a_bit = "1"
        case "A-D":
            comp_bits = "000111"
        case "M-D":
            comp_bits = "000111"
            a_bit = "1"
        case "D&A":
            comp_bits = "000000"
        case "D&M":
            comp_bits = "000000"
            a_bit = "1"
        case "D|A":
            comp_bits = "010101"
        case "D|M":
            comp_bits = "010101"
            a_bit = "1"
        case _:
            raise ValueError("Invalid comp mnemonic.")

    # Determine ddd bits
    match dest:
        case "null":
            dest_bits = "000"
        case "M":
            dest_bits = "001"
        case "D":
            dest_bits = "010"
        case "MD":
            dest_bits = "011"
        case "A":
            dest_bits = "100"
        case "AM":
            dest_bits = "101"
        case "AD":
            dest_bits = "110"
        case "AMD":
            dest_bits = "111"
        case _:
            raise ValueError("Invalid dest mnemonic.")

    # Determine jjj bits
    match jump:
        case "null":
            jump_bits = "000"
        case "JGT":
            jump_bits = "001"
        case "JEQ":
            jump_bits = "010"
        case "JGE":
            jump_bits = "011"
        case "JLT":
            jump_bits = "100"
        case "JNE":
            jump_bits = "101"
        case "JLE":
            jump_bits = "110"
        case "JMP":
            jump_bits = "111"
        case _:
            raise ValueError("Invalid jump mnemonic.")

    out = (
        "111" + a_bit + comp_bits + dest_bits + jump_bits
    )  # C-instruction always starts with 111
    return out


def parse_labels(program: list):
    predefined_symbols = {
        "SCREEN": "16384",
        "KBD": "24576",
        "SP": "0",
        "LCL": "1",
        "ARG": "2",
        "THIS": "3",
        "THAT": "4",
    }
    for i in range(16):  # R0-R15
        predefined_symbols[f"R{i}"] = str(i)
    symbols = {}
    symbol_less_program = []
    user_label_address = 16
    num_jump_labels = 0  # Keeps track of the num of jump labels encountered because they don't count towards line nums
    for lineindex, line in enumerate(program):
        if line[0] != "(":
            continue
        if line[0] == "(" and line[-1] == ")":
            label = line[1:-1]  # Remove the parentheses
            symbols[label] = lineindex - num_jump_labels
            num_jump_labels += 1
    for lineindex, line in enumerate(program):
        if line[0] != "@":
            continue

        if (
            not line[1].isnumeric() and line[1:] not in predefined_symbols
        ):  # non-numeric and not predefined symbol
            label = line[1:]
            if label not in symbols:  # hasn't been recorded yet
                symbols[label] = user_label_address
                user_label_address += 1

    # Second pass
    symbols = dict(
        sorted(symbols.items(), reverse=True, key=lambda item: len(item[0]))
    )  # Ensures that it works with symbols that are substrings of another
    for line in program:
        # HANDLE USER DEFINED SYMBOLS
        for symbol in symbols:
            line = line.replace(symbol, str(symbols[symbol]))

        # HANDLE PREDEFINED SYMBOLS
        if "R" in line:
            for i in range(16):  # R0-R15
                line = line.replace(f"R{i}", str(i))

        for symbol in predefined_symbols:
            line = line.replace(
                symbol, predefined_symbols[symbol]
            )  # Replace each user defined symbol with the corresponding number

        symbol_less_program.append(line)
    symbol_less_program = [
        line for line in symbol_less_program if line[0] != "("
    ]  # Removes all lines with (jump labels)
    return symbol_less_program


def parse_program(program: list):
    machine_program = []
    for line_index, line in enumerate(program):
        if line[0] == "@":  # A-instruction
            machine_program.append(process_a_instruction(line))
        elif line[0] == "(" and line[-1] == ")":
            pass
        else:  # Handle C instruction
            machine_program.append(process_c_instruction(line))
    return machine_program


def assemble(path: str) -> list:
    program = parse_file(path)
    machine_program = parse_program(parse_labels(program))
    return machine_program


def assemble_list(assembly: list[str]) -> list[str]:
    machine_program = parse_program(parse_labels(assembly))
    return machine_program


if __name__ == "__main__":
    path = r"C:\Users\mrjac\Documents\Programming\Jack2Hack\VMtranslator\out.asm"
    assembled = assemble(path)
    # bootstrap_asm = ["@256", "D=A", "@SP", "M=D"]
    # assembled_bootstrap = assemble_list(bootstrap_asm)
    # assembled = bootstrap_asm + assembled
    with open("out.hack", "w") as outfile:
        outfile.writelines(line + "\n" for line in assembled)
