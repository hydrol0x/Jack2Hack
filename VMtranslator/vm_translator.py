from enum import Enum, auto
from pathlib import Path

# push / pop *segment name* <non negative int>


# Stack pointer at address 0 (SP)
# stack starts at 256
# Local segment pointer stored at 1 (LCL)


class TokenType(Enum):
    PUSH = auto()
    POP = auto()

    # Arithmetic/Logic
    ADD = auto()
    SUB = auto()
    NEG = auto()
    EQ = auto()
    GT = auto()
    LT = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    # Numeric memory address
    ADDR = auto()

    # Memory location label
    LABEL = auto()


class TokenError(Exception):
    pass


class LexError(Exception):
    pass


class Lexer:
    def __init__(self, program_file: Path):
        self.program_file = program_file
        self.tokens = []

    def _lex_element(self, element: str) -> TokenType | None:
        match element:
            case "push":
                return TokenType.PUSH
            case "pop":
                return TokenType.POP
            case "add":
                return TokenType.ADD
            case "sub":
                return TokenType.SUB
            case "neg":
                return TokenType.NEG
            case "eq":
                return TokenType.EQ
            case "gt":
                return TokenType.GT
            case "lt":
                return TokenType.LT
            case "and":
                return TokenType.AND
            case "or":
                return TokenType.OR
            case "not":
                return TokenType.NOT
            case _:
                if element.isnumeric():
                    return TokenType.ADDR
                elif element.isalpha():
                    return TokenType.LABEL
                else:
                    raise TokenError(f"Illegal token {element}")

    def _lex_line(self, line):
        elements = line.split(sep="//")[0]  # Throw out comment
        elements = elements.split()  # Split line into individual elements
        lexed_elems: list[TokenType] = []
        for element in elements:
            lexed_elem = self._lex_element(element)
            if lexed_elem:
                lexed_elems.append(lexed_elem)
        return lexed_elems

    def lex_program(self):
        if not self.program_file.suffix == ".vm":
            print(
                "[WARNING] The VM translator is meant to operate on Jack VM code with the extension `.vm`"
            )

        with open(self.program_file, "r") as file:
            program = file.readlines()
            for line in program:
                self.tokens += self._lex_line(line)

    def output(self, output_path: Path):
        with open(output_path, "w") as outfile:
            outfile.writelines([str(token) + "\n" for token in self.tokens])


class Parser:
    program = []

    def __init__(self, program: list[str]):
        self.program = program

    # Go through each line,


if __name__ == "__main__":
    lexer = Lexer(Path(r"./test.vm"))
    lexer.lex_program()
    print(lexer.tokens)
    lexer.output(Path("./out.vm"))
