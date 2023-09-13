tokens = [
    ('NUMBER',   r'\d+'),  # Integer or decimal number
    ('ASSIGN',   r':='),           # Assignment operator
    ('END',      r';'),            # Statement terminator
    ('SEPERATOR',r',')            #seperating comma
    ('STRING',   r'sc|inc\d\d*|dec\d\d*')#possible string things
    ('ID',       r'[A-Za-z]+'),    # Identifiers
    ('OP',       r'[+\-*/]'),      # Arithmetic operators
    ('LPAREN',   r'\(')
    ('RPAREN',   r'\)')
    ('LSQUARE',   r'\[')
    ('RSQUARE',   r'\]')
    ('NEWLINE',  r'\n'),           # Line endings
    ('SKIP',     r'[ \t]+'),       # Skip over spaces and tabs
    ('MISMATCH', r'.'),            #any other char
]