
//--- push constant 5 ---

// D = 5
@5
D=A

// RAM[SP] = D
@SP
A=M
M=D

// SP += 1
@SP
M=M+1

//--- add ---

// SP -=1
@SP
M=M-1

// D = -RAM[SP]
@SP
A=M
D=-M

// RAM[SP] = D
@SP
A=M
M=D

// SP += 1
@SP
M=M+1

//--- push constant 6 ---

// D = 6
@6
D=A

// RAM[SP] = D
@SP
A=M
M=D

// SP += 1
@SP
M=M+1

//--- add ---

// SP -= 1
@SP
M=M-1

// D = RAM[SP]
@SP
A=M
D=M

// SP -= 1
@SP
M=M-1

// RAM[SP] = D+M
@SP
A=M
M=D+M

// SP += 1
@SP
M=M+1

//--- push constant 10 ---

// D = 10
@10
D=A

// RAM[SP] = D
@SP
A=M
M=D

// SP += 1
@SP
M=M+1

//--- add ---

// SP -= 1
@SP
M=M-1

// D = RAM[SP]
@SP
A=M
D=M

// SP -= 1
@SP
M=M-1

// RAM[SP] = M-D
@SP
A=M
M=M-D

// SP += 1
@SP
M=M+1

//--- push constant 9 ---

// D = 9
@9
D=A

// RAM[SP] = D
@SP
A=M
M=D

// SP += 1
@SP
M=M+1

//--- add ---

// SP -= 1
@SP
M=M-1

// D = RAM[SP]
@SP
A=M
D=M

// SP -= 1
@SP
M=M-1

// RAM[SP] = D+M
@SP
A=M
M=D+M

// SP += 1
@SP
M=M+1
