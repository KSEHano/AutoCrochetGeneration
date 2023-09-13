#based on https://github.com/davidcallanan/py-myopl-code episonde 1-3
#######################################
# IMPORTS
#######################################

from strings_with_arrows import *

#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = 'scinde\'\''

#######################################
# ERRORS
#######################################

class Error:
	def __init__(self, pos_start, pos_end, error_name, details):
		self.pos_start = pos_start
		self.pos_end = pos_end
		self.error_name = error_name
		self.details = details
	
	def as_string(self):
		result  = f'{self.error_name}: {self.details}\n'
		result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
		result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
		return result

class IllegalCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
	def __init__(self, pos_start, pos_end, details=''):
		super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
	def __init__(self, pos_start, pos_end, details, context):
		super().__init__(pos_start, pos_end, 'Runtime Error', details)
		self.context = context

	def as_string(self):
		result  = self.generate_traceback()
		result += f'{self.error_name}: {self.details}'
		result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
		return result

	def generate_traceback(self):
		result = ''
		pos = self.pos_start
		ctx = self.context

		while ctx:
			result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Traceback (most recent call last):\n' + result

#######################################
# POSITION
#######################################

class Position:
	def __init__(self, idx, ln, col, fn, ftxt):
		self.idx = idx
		self.ln = ln
		self.col = col
		self.fn = fn
		self.ftxt = ftxt

	def advance(self, current_char=None):
		self.idx += 1
		self.col += 1

		if current_char == '\n':
			self.ln += 1
			self.col = 0

		return self

	def copy(self):
		return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

TT_INT		= 'INT'
TT_FLOAT    = 'FLOAT'
TT_STRING   = 'STRING'
TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_MUL      = 'MUL'
TT_DIV      = 'DIV'
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'
TT_LSQUARE  = 'LSQUARE'
TT_RSQUARE  = 'RSQUARE'
TT_COMMA    = 'COMMA'
TT_QUOTE    = 'QUOTE'
TT_EOF		= 'EOF'

class Token:
	def __init__(self, type_, value=None, pos_start=None, pos_end=None):
		self.type = type_
		self.value = value

		if pos_start:
			self.pos_start = pos_start.copy()
			self.pos_end = pos_start.copy()
			self.pos_end.advance()

		if pos_end:
			self.pos_end = pos_end
	
	def __repr__(self):
		if self.value: return f'{self.type}:{self.value}'
		return f'{self.type}'

#######################################
# LEXER
#######################################

class Lexer:
	def __init__(self, fn, text):
		self.fn = fn
		self.text = text
		self.pos = Position(-1, 0, -1, fn, text)
		self.current_char = None
		self.advance()
	
	def advance(self):
		self.pos.advance(self.current_char)
		self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

	def make_tokens(self):
		tokens = []

		while self.current_char != None:
			if self.current_char in ' \t':
				self.advance()
			elif self.current_char in DIGITS:
				tokens.append(self.make_number())
			elif self.current_char in LETTERS:
				tokens.append(self.make_string())
				self.advance()
			elif self.current_char == '+':
				tokens.append(Token(TT_PLUS, pos_start=self.pos))
				self.advance()
			elif self.current_char == '-':
				tokens.append(Token(TT_MINUS, pos_start=self.pos))
				self.advance()
			elif self.current_char == '*':
				tokens.append(Token(TT_MUL, pos_start=self.pos))
				self.advance()
			elif self.current_char == '/':
				tokens.append(Token(TT_DIV, pos_start=self.pos))
				self.advance()
			elif self.current_char == '(':
				tokens.append(Token(TT_LPAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char == ')':
				tokens.append(Token(TT_RPAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char == '[':
				tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
				self.advance()
			elif self.current_char == ']':
				tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
				self.advance()
			elif self.current_char == ',':
				tokens.append(Token(TT_COMMA, pos_start=self.pos))
				self.advance()
			# elif self.current_char == '\'':
			# 	tokens.append(Token(TT_QUOTE, pos_start=self.pos))
			# 	self.advance()
			else:
				pos_start = self.pos.copy()
				char = self.current_char
				self.advance()
				return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

		tokens.append(Token(TT_EOF, pos_start=self.pos))
		return tokens, None

	def make_number(self):
		num_str = ''
		dot_count = 0
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in DIGITS + '.':
			if self.current_char == '.':
				if dot_count == 1: break
				dot_count += 1
				num_str += '.'
			else:
				num_str += self.current_char
			self.advance()

		if dot_count == 0:
			return Token(TT_INT, int(num_str), pos_start, self.pos)
		else:
			return Token(TT_FLOAT, float(num_str), pos_start, self.pos)
		

		
	def make_string(self):
		str_str = ''
		quote_count = 0
		pos_start = self.pos.copy()
		while self.current_char != None and self.current_char in LETTERS or self.current_char in DIGITS:
			if self.current_char == '\'':
				if quote_count == 1: break
				quote_count += 1
				
			else:
				str_str += self.current_char
			self.advance()

		
		return Token(TT_STRING, str_str, pos_start, self.pos)



#######################################
# NODES
#######################################

class NumberNode:
	def __init__(self, tok):
		self.tok = tok

		self.pos_start = self.tok.pos_start
		self.pos_end = self.tok.pos_end

	def __repr__(self):
		return f'{self.tok}'
	
class StitchNode:
	def __init__(self, fac_tok, lit_token) -> None:
		self.fac = fac_tok
		self.lit = lit_token

		self.pos_start = self.fac.pos_start
		self.pos_end = self.lit.pos_end
	
	def __repr__(self) -> str:
		return f'{self.fac} {self.lit}'
	
class FactorNode:
	def __init__(self, tok):
		self.fac = tok

		self.pos_start = self.fac.pos_start
		self.pos_end = self.fac.pos_end

	def __repr__(self):
		return f'{self.fac}'

	

class BinOpNode:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

		self.pos_start = self.left_node.pos_start
		self.pos_end = self.right_node.pos_end

	def __repr__(self):
		return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

		self.pos_start = self.op_tok.pos_start
		self.pos_end = node.pos_end

	def __repr__(self):
		return f'({self.op_tok}, {self.node})'

#######################################
# PARSE RESULT
#######################################

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None

	def register(self, res):
		if isinstance(res, ParseResult):
			if res.error: self.error = res.error
			return res.node

		return res

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		self.error = error
		return self

#######################################
# PARSER
#######################################

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tok_idx = -1
		self.advance()

	def advance(self, ):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok
		
	
	def backup(self,):
		self.tok_idx -= 1
		if self.tok_idx > -1:
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok

	def parse(self):
		res = self.summing()
		if not res.error and self.current_tok.type != TT_EOF:
			
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected of form [(INT, 'STRING'),....]"
			))
		return res

	###################################

	# def factor(self):
	# 	res = ParseResult()
	# 	tok = self.current_tok

	# 	if tok.type in (TT_PLUS, TT_MINUS):
	# 		res.register(self.advance())
	# 		factor = res.register(self.factor())
	# 		if res.error: return res
	# 		return res.success(UnaryOpNode(tok, factor))
		
	# 	elif tok.type in (TT_INT, TT_FLOAT):
	# 		res.register(self.advance())
	# 		return res.success(NumberNode(tok))

	# 	elif tok.type == TT_LPAREN:
	# 		res.register(self.advance())
	# 		expr = res.register(self.expr())
	# 		if res.error: return res
	# 		if self.current_tok.type == TT_RPAREN:
	# 			res.register(self.advance())
	# 			return res.success(expr)
	# 		else:
	# 			return res.failure(InvalidSyntaxError(
	# 				self.current_tok.pos_start, self.current_tok.pos_end,
	# 				"Expected ')'"
	# 			))

	# 	return res.failure(InvalidSyntaxError(
	# 		tok.pos_start, tok.pos_end,
	# 		"Expected int or float"
	# 	))

	def stitch(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type == TT_LPAREN:
			res.register(self.advance())
			if self.current_tok.type == TT_INT:
				sum = res.register(self.stitch())
			else:
				sum = res.register(self.summing())
			if res.error: return res
			if self.current_tok.type == TT_RPAREN:
				res.register(self.advance())
				return res.success(sum)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ')'"
				))
		elif tok.type == TT_LSQUARE:
			res.register(self.advance())
			sum = res.register(self.summing())
			if res.error: return res
			if self.current_tok.type == TT_RSQUARE:
				res.register(self.advance())
				return res.success(sum)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ']'"
				))
		
		elif tok.type == TT_INT:
			fac = tok
			res.register(self.advance())#comma
			if self.current_tok.type == TT_COMMA:
				res.register(self.advance())
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ','"
					))

			
			if self.current_tok.type == TT_STRING:
				lit = self.current_tok
				res.register(self.advance())
				if res.error: return res
				return res.success(StitchNode(fac, lit))
			
			elif self.current_tok.type in (TT_LPAREN, TT_LSQUARE): 
				res.register(self.backup())
				res.register(self.backup())
				mult = res.register(self.multing())
				if res.error: return res
				return res.success(mult)
			
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected 'STRING' or (Expresion)"
					))
				
		return res.failure(InvalidSyntaxError(
			tok.pos_start, tok.pos_end,
			"Expected '(INT, 'STRING')'"
		))
	
	def factor(self):
		res = ParseResult()
		tok = self.current_tok
		res.register(self.advance())
		
		return res.success(FactorNode(tok))

	def summing(self):
		return self.bin_op(self.stitch, self.stitch, TT_PLUS)
	
	def multing(self):
		
		return  self.bin_op(self.factor, self.stitch, TT_MUL)
		# if result:
		# 	return result
		# else:
		# 	return self.summing()



	# def term(self):
	# 	return self.bin_op(self.factor, (TT_MUL, TT_DIV))

	# def expr(self):
	# 	return self.bin_op(self.term, (TT_PLUS, TT_MINUS))
	

	##################################


	def bin_op(self, func1, func2, ops):
		res = ParseResult()
		left = res.register(func1())
		if res.error: return res
		
		while self.current_tok.type == TT_COMMA:
			op_tok = ops
			
			res.register(self.advance())
			right = res.register(func2())
			if res.error: return res
			left = BinOpNode(left, op_tok, right)
		
		return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
	def __init__(self):
		self.value = None
		self.error = None

	def register(self, res):
		if res.error: self.error = res.error
		return res.value

	def success(self, value):
		self.value = value
		return self

	def failure(self, error):
		self.error = error
		return self

#######################################
# VALUES
#######################################

# class Number:
# 	def __init__(self, value):
# 		self.value = value
# 		self.set_pos()
# 		self.set_context()

# 	def set_pos(self, pos_start=None, pos_end=None):
# 		self.pos_start = pos_start
# 		self.pos_end = pos_end
# 		return self

# 	def set_context(self, context=None):
# 		self.context = context
# 		return self

# 	def added_to(self, other):
# 		if isinstance(other, Number):
# 			return Number(self.value + other.value).set_context(self.context), None

# 	def subbed_by(self, other):
# 		if isinstance(other, Number):
# 			return Number(self.value - other.value).set_context(self.context), None

# 	def multed_by(self, other):
# 		if isinstance(other, Number):
# 			return Number(self.value * other.value).set_context(self.context), None

# 	def dived_by(self, other):
# 		if isinstance(other, Number):
# 			if other.value == 0:
# 				return None, RTError(
# 					other.pos_start, other.pos_end,
# 					'Division by zero',
# 					self.context
# 				)

# 			return Number(self.value / other.value).set_context(self.context), None

# 	def __repr__(self):
# 		return str(self.value)


class Stitch:

	def __init__(self, fac, lit):
		self.fac = fac
		self.lit = lit
		self.set_pos()
		self.set_context()

	def __eq__(self, other):
		if isinstance(other, Stitch):
			return (self.fac == other.fac and self.lit == other.lit)
		return False

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self
	
	def add_to(self, other):
		if isinstance(other, Stitch) and other.lit == self.lit:
			return Stitch(self.fac + other.fac, self.lit).set_context(self.context), None
		elif isinstance(other, Stitch) and other.lit != self.lit:
			return StitchList([self, other]).set_context(self.context), None

		elif isinstance(other, StitchList):
			
			if other.fac:
				
				return StitchList([self, other]).set_context(self.context), None
			
			if isinstance(other.slist[0], StitchList):
				result, error = self.add_to(other.slist[0])
				return StitchList([result, *other.slist[1:]]), error
			if self.lit == other.slist[0].lit :
				result, error = self.add_to(other.slist[0])
				return StitchList([result, *other.slist[1:]]).set_context(self.context), error
		
			else:
				return StitchList([self, other]).set_context(self.context), None

		
	def mult_by(self, other):
		if isinstance(other, Factor):
			return Stitch(self.fac * other.fac, self.lit).set_context(self.context), None

	
	def __repr__(self):
		if self.fac > 1:
			return f'{self.fac} {self.lit}'
		else:
			return f'{self.lit}'
		
class Factor:
	def __init__(self, fac):
		self.fac = fac
		self.set_pos()
		self.set_context()
	
	def __eq__(self, other):
		if isinstance(other, Factor):
			return self.fac == other.fac
		return False

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self
	
	def __repr__(self):
		return f'{self.fac}'
	
class StitchList:
	def __init__(self, slist, fac = None):
		self.slist = slist
		self.fac = fac
		

		self.set_pos()
		self.set_context()

	def __eq__(self, other):
		if isinstance(other, StitchList):
			return (self.slist == other.slist and self.fac == other.fac)
		return False

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self	
	
	def add_to(self, other):		
		
		if len(self.slist) > 0 and isinstance(self.slist[-1], StitchList) and self.fac == None:
			
			result, error = self.slist[-1].add_to(other)
			return StitchList([*self.slist[:-1], result]).set_context(self.context), None

		if isinstance(other, Stitch) :
			if self.fac:
				return StitchList([self, other]), None
									
			if self.slist[-1].lit == other.lit: # and self.fac == None:
				result, error = self.slist[-1].add_to(other)
				return StitchList([*self.slist[:-1], result]).set_context(self.context), None
			
			else:# and self.fac == None:
				return StitchList([*self.slist, other]).set_context(self.context), None
			
			
		
		
		elif isinstance(other, StitchList):
			
				
			if isinstance(self.slist[-1], Stitch) and isinstance(other.slist[0], Stitch) and  other.fac == None and self.fac == None:
				result, error = self.slist[-1].add_to(other.slist[0])
				return StitchList([*self.slist[:-1], result, *other.slist[1:]]).set_context(self.context), None

			return StitchList([self, other]).set_context(self.context),None
		
			 
		
	def mult_by(self, other):

		if isinstance(other, Factor):

			if self.fac:
				return StitchList(self.slist, fac = self.fac*other.fac).set_context(self.context), None
			else:
				self.fac = other.fac
				return self.set_context(self.context), None
		
	def __repr__(self):
		
		def make_repr():
			repr = f''
			
			for value in self.slist:
				
				if isinstance (value, Stitch):
					
					repr += f' {value},'
				elif isinstance(value, StitchList):
					repr += f' {value},'
			return repr[:-1]

		if self.fac:
			repr = f'{self.fac}({make_repr()})'
		
		else:
			repr = f'{make_repr()}'
			
		
		return repr
		
		
#######################################
# CONTEXT
#######################################

class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos

#######################################
# INTERPRETER
#######################################

class Interpreter:
	def visit(self, node, context):
		method_name = f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)

	def no_visit_method(self, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined')

	###################################

	# def visit_NumberNode(self, node, context):
	# 	return RTResult().success(
	# 		Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
	# 	)

	def visit_StitchNode(self, node, context):
		return RTResult().success(
			Stitch(node.fac.value, node.lit.value ).set_context(context).set_pos(node.pos_start, node.pos_end)
		)
	
	def visit_FactorNode(self, node, context):
		return RTResult().success(
			Factor(node.fac.value).set_context(context).set_pos(node.pos_start, node.pos_end)
		)

	def visit_BinOpNode(self, node, context):
		res = RTResult()
		left = res.register(self.visit(node.left_node, context))
		if res.error: return res
		right = res.register(self.visit(node.right_node, context))
		if res.error: return res

		if node.op_tok == TT_PLUS:
			result, error = left.add_to(right)
		elif node.op_tok == TT_MUL:
			result, error = right.mult_by(left)
		
		if error:
			return res.failure(error)
		else:
			return res.success(result.set_pos(node.pos_start, node.pos_end))

	def visit_UnaryOpNode(self, node, context):
		res = RTResult()
		number = res.register(self.visit(node.node, context))
		if res.error: return res

		error = None

		# if node.op_tok.type == TT_MINUS:
		# 	number, error = number.multed_by(Number(-1))

		if error:
			return res.failure(error)
		else:
			return res.success(number.set_pos(node.pos_start, node.pos_end))

#######################################
# RUN
#######################################

def run(fn, text):
	# Generate tokens
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_tokens()
	if error: return None, error
	
	# Generate AST
	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error
	
	#print(ast.node)
	# # Run program
	interpreter = Interpreter()
	context = Context('<program>')
	result = interpreter.visit(ast.node, context)

	return result.value, result.error #ast.node, ast.error