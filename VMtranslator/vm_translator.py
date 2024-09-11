from enum import Enum, auto
from pathlib import Path

# push / pop *segment name* <non negative int>


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

    # Symbol
    SYM = auto()

    # Branching
    LABEL = auto()
    GOTO = auto()
    IFGT = auto()

    # Used for parsing; not part of specification
    EOF = auto()

    def __str__(self):
        return self.name


class TokenError(Exception):
    pass


class Token:
    def __init__(self, type_: TokenType, lexeme: str, literal: object = None):
        self.type = type_
        self.lexeme = lexeme  # text representation, like `push`, `pop`, `add`
        self.literal = literal  # Pyton object representing any literal values

    def __repr__(self) -> str:
        return f"⧙{self.type} '{self.lexeme}'⧘"


class Symbol:
    def __init__(self, token: Token, pointer: int):
        # offset is start of memory segment in RAM
        self.token = token
        self.pointer = pointer

    def __repr__(self) -> str:
        return str(self.token)


class LexError(Exception):
    pass


class Lexer:
    def __init__(self, program):
        self.program: list[str] = program
        # self.program_file = program_file
        self.tokens: list[Token] = []

    def _lex_element(self, element: str) -> Token | None:
        match element:
            case "push":
                return Token(TokenType.PUSH, element)
            case "pop":
                return Token(TokenType.POP, element)
            case "add":
                return Token(TokenType.ADD, element)
            case "sub":
                return Token(TokenType.SUB, element)
            case "neg":
                return Token(TokenType.NEG, element)
            case "eq":
                return Token(TokenType.EQ, element)
            case "gt":
                return Token(TokenType.GT, element)
            case "lt":
                return Token(TokenType.LT, element)
            case "and":
                return Token(TokenType.AND, element)
            case "or":
                return Token(TokenType.OR, element)
            case "not":
                return Token(TokenType.NOT, element)
            case "label":
                return Token(TokenType.LABEL, element)
            case "goto":
                return Token(TokenType.GOTO, element)
            case "if-goto":
                return Token(TokenType.IFGT, element)
            case _:
                if element.isnumeric():
                    return Token(TokenType.ADDR, element, int(element))
                elif element.isalpha():
                    return Token(TokenType.SYM, element, element)
                else:
                    raise TokenError(f"Illegal token {element}")

    def _lex_line(self, line):
        elements = line.split(sep="//")[0]  # Throw out comment
        elements = elements.split()  # Split line into individual elements
        lexed_elems: list[Token] = []
        for element in elements:
            lexed_elem = self._lex_element(element)
            if lexed_elem:
                lexed_elems.append(lexed_elem)
        return lexed_elems

    def read_file(self, path: Path):
        if not path.suffix == ".vm":
            print(
                "[WARNING] The VM translator is meant to operate on Jack VM code with the extension `.vm`"
            )

        with open(path, "r") as file:
            self.program = file.readlines()

    def lex_program(self):
        for line in self.program:
            self.tokens += self._lex_line(line)
        self.tokens.append(Token(TokenType.EOF, "EOF"))

        return self.tokens

    def output(self, output_path: Path):
        with open(output_path, "w") as outfile:
            outfile.writelines([str(token) + "\n" for token in self.tokens])


class Statement:
    pass


class PushStatement(Statement):
    def __init__(self, symbol: Symbol, address: Token):
        # push label address
        self.symbol = symbol
        self.address = address

    def __repr__(self) -> str:
        return f"PUSH({self.symbol}, {self.address})"


class PopStatement(Statement):
    def __init__(self, symbol: Symbol, address: Token):
        # push label address
        self.symbol = symbol
        self.address = address

    def __repr__(self) -> str:
        return f"PUSH({self.symbol}, {self.address})"


class ArithmeticStatement(Statement):
    def __init__(self, op_type: TokenType):
        self.op_type = op_type

    def __repr__(self) -> str:
        return f"{self.op_type.name}"


class LogicStatement(Statement):
    def __init__(self, op_type: TokenType):
        self.op_type = op_type


class ParseError(Exception):
    pass


# Will return list of valid operations
class Parser:
    program = []

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.current = 0
        self.program: list[Token] = []
        # Stack pointer at address 0 (SP)
        # stack starts at 256
        # Local segment pointer stored at 1 (LCL)
        # R<N> is memory[13-15]
        # Static from X.vm is X.i for a symbol i
        self.SP: int = 0
        self.LOCAL_PTR: int = 1
        self.ARG_PTR: int = 2
        self.THIS_PTR: int = 3
        self.THAT_PTR: int = 4
        self.STACK_PTR: int = 256
        self.TEMP_PTR: int = 5

    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]

    def advance(self):
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def match(self, *t_types: TokenType) -> bool:
        for t_type in t_types:
            if self.check(t_type):
                self.advance()
                return True
        return False

    def previous(self) -> Token:
        return self.tokens[self.current - 1]

    def check(self, expected: TokenType) -> bool:
        if self.is_at_end():
            return False
        else:
            return self.peek().type == expected

    def consume(self, type: TokenType):
        if self.check(type):
            return self.advance()

        token = self.peek()
        raise ParseError(f"Unexpected token {token}")

    def get_symbol(self, token: Token) -> Symbol:
        offset = 0
        match token.lexeme:
            case "local":
                offset = self.LOCAL_PTR
            case "argument":
                offset = self.ARG_PTR
            case "this":
                offset = self.THIS_PTR
            case "that":
                offset = self.THAT_PTR
            case "constant":
                offset = 0
                # Const will push a value to the stack
                # const is not a real segment, simply mem[SP] = i
            case "temp":
                offset = self.TEMP_PTR
            case "pointer":
                # Pointer will return THIS if 0 and THAT if 1
                offset = 0

        return Symbol(token, offset)

    def parse(self) -> list[Statement]:
        statements = []
        while not self.is_at_end():
            token = self.peek()
            if self.match(TokenType.PUSH):
                symbol_token = self.consume(TokenType.SYM)
                symbol = self.get_symbol(symbol_token)
                address = self.consume(TokenType.ADDR)
                statements.append(PushStatement(symbol, address))
                continue
            elif self.match(TokenType.POP):
                symbol_token = self.consume(TokenType.SYM)
                symbol = self.get_symbol(symbol_token)
                address = self.consume(TokenType.ADDR)
                statements.append(PopStatement(symbol, address))
                continue
            elif self.match(TokenType.ADD):
                statements.append(ArithmeticStatement(TokenType.ADD))
                continue
            elif self.match(TokenType.SUB):
                statements.append(ArithmeticStatement(TokenType.SUB))
                continue
            elif self.match(TokenType.NEG):
                statements.append(ArithmeticStatement(TokenType.NEG))
                continue
            elif self.match(TokenType.EQ):
                statements.append(ArithmeticStatement(TokenType.EQ))
                continue
            elif self.match(TokenType.GT):
                statements.append(ArithmeticStatement(TokenType.GT))
                continue
            elif self.match(TokenType.LT):
                statements.append(ArithmeticStatement(TokenType.LT))
                continue
            elif self.match(TokenType.AND):
                statements.append(LogicStatement(TokenType.AND))
                continue
            elif self.match(TokenType.OR):
                statements.append(LogicStatement(TokenType.OR))
                continue
            elif self.match(TokenType.NOT):
                statements.append(LogicStatement(TokenType.NOT))
                continue
            elif self.match(TokenType.ADDR):
                raise ParseError(
                    f"Expected command or Arithmetic/Logic operation, found address {self.previous()}.\n Address can only come with `push`, `pop`, `function`, `call` commands"
                )
            elif self.match(TokenType.SYM):
                raise ParseError(
                    f"Expected command or Arithmetic/Logic operation, found symbol {self.previous()}.\n Symbol can only come with `push`, `goto`, `if-goto`, `function`, `call` commands"
                )
            else:
                raise ParseError(f"Illegal token {self.peek()}")
        return statements


class WriterError(Exception):
    pass


class CodeWriter:
    # TODO: unhardcode the offsets for memory segments, access through ram pointers
    # Writes ASM from program parse
    def __init__(self, parsed_program: list[Statement]):
        self.program = parsed_program

    def push_asm(self, push: PushStatement) -> list[str]:
        # TODO: implement push and pop to static
        symbol = push.symbol
        address = int(push.address.lexeme)
        if symbol.token.lexeme == "constant":
            # Push `address` to stack
            # Here, address is not an address but just a constant/litearl value
            asm = [
                f"\n// D = {address}",
                f"@{address}",
                "D=A",
                "\n// RAM[SP] = D",
                "@SP",
                "A=M",
                "M=D",
                "\n// SP += 1",
                "@SP",
                "M=M+1",
            ]
        elif symbol.token.lexeme == "pointer":
            # Pointer will return THIS if 0 and THAT if 1
            # push to this if 0, push to that if 1
            if address == 0:
                # *SP = this
                asm = [
                    "\n// D = this",
                    "@THIS",
                    "D=A",
                    "\n// RAM[SP] = THIS",
                    "@SP",
                    "A=M",
                    "M=D",
                    "\n// SP += 1",
                    "@SP",
                    "M=M+1",
                ]
            elif address == 1:
                # *SP = that
                asm = [
                    "\n// D = this",
                    "@THAT",
                    "D=A",
                    "\n// RAM[SP] = THIS",
                    "@SP",
                    "A=M",
                    "M=D",
                    "\n// SP += 1",
                    "@SP",
                    "M=M+1",
                ]
            else:
                raise WriterError(
                    f"Invalid value pushed to `pointer`. Expected `0` or `1` found {address}"
                )
        else:
            asm = [
                f"\n// D = {address}+RAM[{symbol.pointer}]",
                f"@{symbol.pointer}",
                "D=M",
                f"@{address}",
                "D=D + A",  # D is the address needed
                f"\n// D = RAM[D]",
                "A=D",
                "D=M",
                f"\n// RAM[SP] = {symbol.token.lexeme}[{address}]",
                "@SP",
                "A=M",
                "M=D",
                "\n// SP += 1",
                "@SP",
                "M=M+1",
            ]

        # Push value at symbol[address] to stack
        # Converted to mem[address + symbol_start]
        return asm

    def pop_asm(self, pop: PopStatement) -> list[str]:
        # TODO: properly handl *SP shoudl be @SP, A=M
        symbol = pop.symbol
        address = int(pop.address.lexeme)
        if symbol.token.lexeme == "constant":
            # Push `address` to stack
            # Here, address is not an address but just a constant/litearl value
            raise WriterError("Illegal argument. Cannot pop from `constant`")
        elif symbol.token.lexeme == "pointer":
            # Pointer will return THIS if 0 and THAT if 1
            # push to this if 0, push to that if 1
            this_that = ""
            if address == 0:
                # *SP = this
                this_that = "THIS"
            elif address == 1:
                # *SP = that
                this_that = "THAT"
            else:
                raise WriterError(
                    f"Invalid value pushed to `pointer`. Expected `0` or `1` found {address}"
                )
            asm = [
                "\n// D = RAM[SP]",
                "@SP",
                "A=M",
                "D=M",
                f"\n// THIS = D",
                f"@{this_that}",
                "M=D",
                "\n // SP -= 1",
                "@SP",
                "M=M-1",
            ]
        else:
            asm = [
                f"\n//R13 = RAM[{symbol.token.lexeme}] + {address}",
                f"@{symbol.pointer}",
                "D=M",
                f"@{address}",
                "D=D+M",
                "@R13",
                "M=D",
                "\n// D = RAM[SP]",
                "@SP",
                "A=M",
                "D=M",
                "\n// RAM[R13] = D" "@R13",
                "A=M",
                "M=D",
                "\n // SP -= 1",
                "@SP",
                "M=M-1",
            ]

        # Push value at symbol[address] to stack
        # Converted to mem[address + symbol_start]
        return asm

    def arithmetic_asm(self, statement: ArithmeticStatement) -> list[str]:
        asm = []
        match statement.op_type:
            case TokenType.ADD:
                asm = [
                    "\n// SP -= 1",
                    "@SP",
                    "M=M-1",
                    "\n// D = RAM[SP]",
                    "@SP",
                    "A=M",
                    "D=M",
                    "\n// SP -= 1",
                    "@SP",
                    "M=M-1",
                    "\n// RAM[SP] = D+M",
                    "@SP",
                    "A=M",
                    "M=D+M",
                    "\n// SP += 1",
                    "@SP",
                    "M=M+1",
                ]
            case TokenType.SUB:
                asm = [
                    "\n// SP -= 1",
                    "@SP",
                    "M=M-1",
                    "\n// D = RAM[SP]",
                    "@SP",
                    "A=M",
                    "D=M",
                    "\n// SP -= 1",
                    "@SP",
                    "M=M-1",
                    "\n// RAM[SP] = M-D",
                    "@SP",
                    "A=M",
                    "M=M-D",
                    "\n// SP += 1",
                    "@SP",
                    "M=M+1",
                ]
        return asm

    # def logic_asm(self, statement) -> list[str]:
    #     pass

    def write_code(self):
        out_asm = []
        for statement in self.program:
            if isinstance(statement, PushStatement):
                descriptor = [
                    f"\n//--- push {statement.symbol.token.lexeme} {statement.address.lexeme} ---"
                ]
                asm = self.push_asm(statement)
                descriptor.extend(asm)
                out_asm += descriptor
            elif isinstance(statement, PopStatement):
                descriptor = [
                    f"\n//--- pop {statement.symbol.token.lexeme} {statement.address.lexeme} ---"
                ]
                asm = self.pop_asm(statement)
                descriptor.extend(asm)
                out_asm += descriptor
            elif isinstance(statement, ArithmeticStatement):
                descriptor = [f"\n//--- add ---"]
                asm = self.arithmetic_asm(statement)
                descriptor.extend(asm)
                out_asm += descriptor
            elif isinstance(statement, LogicStatement):
                raise NotImplementedError
                self.logic_asm(statement)

        return out_asm


if __name__ == "__main__":
    program = [
        "push this 7",
        "push constant 13",
        "push local 20",
        "pop local 20",
        "pop local 22",
    ]  # "push local 12", "pop constant 23", "add"]
    lexer = Lexer(program)
    lexer.read_file(
        Path(
            r"C:\Users\mrjac\Documents\Programming\Jack2Hack\VMtranslator\StackArithmetic\SimpleAdd\SimpleAdd.vm"
        )
    )
    lexed = lexer.lex_program()
    print("Lexed Program")
    print(*lexed, sep="\n")
    parser = Parser(lexed)
    print("Parsed Program")
    parsed = parser.parse()
    print(*parsed, sep="\n")

    writer = CodeWriter(parsed)
    print(*writer.write_code(), sep="\n")
    with open("./out.asm", "w") as outfile:
        outfile.writelines([line + "\n" for line in writer.write_code()])
    # print(*[line.lstrip(" ") + "\n" for line in writer.push_asm(parsed[0])], sep="")
