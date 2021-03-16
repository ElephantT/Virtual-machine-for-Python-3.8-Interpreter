"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import collections
import dis
import types
import typing as tp
import operator


class VMError(Exception):
    """For raising errors in the operation of the VM."""
    pass


Block = collections.namedtuple("Block", "type, handler, level")


def bind_args(arg_defaults: tp.Any, arg_kwdefaults: tp.Any, code: tp.Any, *args: tp.Any, **kwargs: tp.Any) \
        -> tp.Dict[str, tp.Any]:
    """Bind values from `args` and `kwargs` to corresponding arguments of `func`

    :param code:
    :param arg_kwdefaults:
    :param arg_defaults:
    :param func: function to be inspected
    :param args: positional arguments to be bound
    :param kwargs: keyword arguments to be bound
    :return: `dict[argument_name] = argument_value` if binding was successful,
             raise TypeError with one of `ERR_*` error descriptions otherwise
    """
    CO_VARARGS = 4
    CO_VARKEYWORDS = 8
    ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
    ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
    ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
    ERR_MISSING_POS_ARGS = 'Missing positional arguments'
    ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
    ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'

    # checker, if there is enough parameters
    if not bool(code.co_flags & CO_VARKEYWORDS):
        for i in kwargs.keys():
            if i in code.co_varnames[:code.co_posonlyargcount]:
                raise TypeError(ERR_POSONLY_PASSED_AS_KW)

    if code.co_flags & CO_VARARGS == 0 and len(args) > code.co_argcount:
        raise TypeError(ERR_TOO_MANY_POS_ARGS)
    if code.co_flags & CO_VARKEYWORDS == 0:
        for i in kwargs.keys():
            if i not in code.co_varnames:
                raise TypeError(ERR_TOO_MANY_KW_ARGS)

    answer: tp.Dict[str, tp.Any]
    answer = {}
    args_users_name = 'args'
    kwargs_users_name = 'kwargs'
    all_sized_args = code.co_argcount + code.co_kwonlyargcount

    if code.co_flags & CO_VARARGS != 0:
        args_users_name = code.co_varnames[all_sized_args]
        answer[args_users_name] = tuple()
    if code.co_flags & CO_VARKEYWORDS != 0:
        if code.co_flags & CO_VARARGS != 0:
            args_users_name = code.co_varnames[all_sized_args]
            kwargs_users_name = code.co_varnames[all_sized_args + 1]
        else:
            kwargs_users_name = code.co_varnames[all_sized_args]
        answer[kwargs_users_name] = dict()

    # все чиселки слева занесли, если их слишком много, то остаток в аргс
    index = 0
    for argument in args:
        if index == code.co_argcount and index != len(args):
            answer[args_users_name] = tuple(args[index:])
            break
        answer[code.co_varnames[index]] = argument
        index += 1
    # если чиселками до аргс мы не заполнили, то проверяем, какие из аргс потенциальных (что между чертой и кваргс)
    # перешли в кваргс
    args_to_kwargs_number = 0  # кол-во перешедших в кваргс
    defaults_used = 0  # кол-во использованных дефолтов
    j_index = 0  # скок элементов прошли от границы слева от *args
    while index < code.co_argcount - j_index:
        if code.co_varnames[code.co_argcount - 1 - j_index] in kwargs.keys():
            if code.co_argcount - j_index > code.co_posonlyargcount:
                assert code.co_varnames[code.co_argcount - 1 - j_index] not in answer.keys()
                answer[code.co_varnames[code.co_argcount - 1 - j_index]] = \
                    kwargs[code.co_varnames[code.co_argcount - 1 - j_index]]
                args_to_kwargs_number += 1
            else:
                if code.co_flags & CO_VARKEYWORDS == 0:
                    answer[kwargs_users_name][code.co_varnames[code.co_argcount - 1 - j_index]] = \
                        kwargs[code.co_varnames[code.co_argcount - 1 - j_index]]
                else:
                    raise TypeError(ERR_MISSING_POS_ARGS)
        else:
            if not arg_defaults:
                raise TypeError(ERR_MISSING_POS_ARGS)
            if len(arg_defaults) - 1 - j_index < 0:
                raise TypeError(ERR_MISSING_POS_ARGS)
            answer[code.co_varnames[code.co_argcount - 1 - j_index]] = \
                arg_defaults[len(arg_defaults) - 1 - j_index]
            defaults_used += 1
        j_index += 1

    #  ERR_MULT_VALUES_FOR_ARG видимо она работает ток для тех, кто между / и *args, и при том они есть в kwargs, а
    #  **kwargs-а нет
    for idx in range(max(code.co_posonlyargcount - 1, 0), index - 1):
        if code.co_varnames[idx] in kwargs.keys():
            if not bool(code.co_flags & CO_VARKEYWORDS):
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)
            else:
                answer[kwargs_users_name][code.co_varnames[idx]] = kwargs[code.co_varnames[idx]]

    #  теперь все из kwargs занесены к нам
    for kwarg in kwargs.keys():
        if kwargs_users_name in answer.keys():
            if kwarg in answer[kwargs_users_name].keys():
                continue
            else:
                if kwarg not in code.co_varnames[code.co_posonlyargcount:all_sized_args]:
                    answer[kwargs_users_name][kwarg] = kwargs[kwarg]
        if kwarg in answer.keys():
            continue
        else:
            x = code.co_argcount
            if kwarg in code.co_varnames[x: code.co_kwonlyargcount + x]:
                answer[kwarg] = kwargs[kwarg]
            else:
                answer[kwargs_users_name][kwarg] = kwargs[kwarg]

    #  добили дефолтными значениями
    if arg_kwdefaults:
        for default_kwarg in arg_kwdefaults.keys():
            if default_kwarg in answer.keys():
                continue
            else:
                answer[default_kwarg] = arg_kwdefaults[default_kwarg]

    # последний тест на сравнение, пробили ли мы все ключи в answer, если нет, то ошибка, нам не хватило kwargs
    for each_arg in code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]:
        if each_arg not in answer.keys():
            if kwargs_users_name in answer.keys():
                if each_arg not in answer[kwargs_users_name].keys():
                    raise TypeError(ERR_MISSING_KWONLY_ARGS)
                else:
                    continue
            if args_users_name in answer.keys():
                if each_arg not in answer[args_users_name]:
                    raise TypeError(ERR_MISSING_KWONLY_ARGS)
                else:
                    continue
            raise TypeError(ERR_MISSING_KWONLY_ARGS)

    return answer


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.6/Include/frameobject.h#L17

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: tp.Dict[str, tp.Any],
                 frame_globals: tp.Dict[str, tp.Any],
                 frame_locals: tp.Dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.f_lasti = 0
        self.last_exception: tp.Tuple[tp.Any, tp.Any, tp.Any]
        self.arg_extended = False
        self.block_stack = []

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def peek(self, n: int) -> tp.Any:
        """Get a value `n` entries down in the stack, without changing the stack."""
        return self.data_stack[-n]

    def jump(self, jump: int) -> None:
        """Move the bytecode pointer to `jump`, so it will execute next."""
        self.f_lasti = jump - 2

    def run(self) -> tp.Any:
        all_instructions_from_dis = list(dis.get_instructions(self.code))
        while self.f_lasti < 2 * len(list(all_instructions_from_dis)):
            # print(self.f_lasti)
            instruction = all_instructions_from_dis[self.f_lasti // 2]
            # argument = self.parse_argument(instruction.opname, instruction.argval, what_to_execute)
            if instruction.opname.startswith('UNARY_'):
                self.unaryOperator(instruction.opname[6:])
            elif instruction.opname.startswith('BINARY_'):
                self.binaryOperator(instruction.opname[7:])
            elif instruction.opname.startswith('INPLACE_'):
                self.inplaceOperator(instruction.opname[8:])
            # elif 'SLICE+' in instruction.opname:
            #     self.sliceOperator(instruction.opname)
            else:
                # dispatch
                if hasattr(self, instruction.opname.lower() + "_op"):
                    getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
                else:
                    raise ModuleNotFoundError(
                        "unknown bytecode type: %s" % instruction.opname
                    )
            self.f_lasti += 2
        # print(len(list(all_instructions_from_dis)) * 2)

        return self.return_value

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-LOAD_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L2416
        """
        # TODO: parse all scopes
        if arg in self.locals.keys():
            self.push(self.locals[arg])
        elif arg in self.globals.keys():
            self.push(self.globals[arg])
        elif arg in self.builtins.keys():
            self.push(self.builtins[arg])
        else:
            raise NameError("name '%s' is not defined" % arg)

    def delete_name_op(self, name: str) -> None:
        del self.locals[name]

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-LOAD_GLOBAL

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L2480
        """
        # TODO: parse all scopes
        # self.push(self.builtins[arg])
        if arg in self.globals:
            val = self.globals[arg]
        elif arg in self.builtins:
            val = self.builtins[arg]
        else:
            raise NameError("global name '%s' is not defined" % arg)
        self.push(val)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-LOAD_CONST

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L1346
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-RETURN_VALUE

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L1911
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-POP_TOP

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L1361
        """
        self.pop()

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-MAKE_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L3571

        Parse stack:
            https://github.com/python/cpython/blob/3.8/Objects/call.c#L671

        Call function in cpython:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L4950
        """
        name = self.pop()  # the qualified name of the function (at TOS)  # noqa
        code = self.pop()  # the code associated with the function (at TOS1)

        defaults = {}
        kwdefaults = {}
        if arg & 8 == 8:
            # a tuple containing cells for free variables, making a closure
            self.pop()
        if arg & 4 == 4:
            #  annotation dict
            self.pop()
        if arg & 2 == 2:
            kwdefaults = self.pop()
        if arg & 1 == 1:
            defaults = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: tp.Dict[str, tp.Any] = {}
            parsed_args = bind_args(defaults, kwdefaults, code, *args, **kwargs)

            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def call_function_op(self, arg: int) -> None:
        """
        Operation description:
                https://docs.python.org/release/3.8.5/library/dis.html#opcode-CALL_FUNCTION

            Operation realization:
                https://github.com/python/cpython/blob/3.8/Python/ceval.c#L3496
        """
        arguments = self.popn(arg)
        f = self.pop()
        self.push(f(*arguments))

    def call_function_kw_op(self, arg: int) -> None:
        kwargs_keys = self.pop()
        values = self.popn(arg)
        args_len = arg - len(kwargs_keys)
        f = self.pop()
        kwargs: tp.Dict[tp.Any, tp.Any]
        kwargs = {}
        args: tp.Tuple[tp.Any, ...]
        args = tuple(values[:args_len])
        for idx in range(len(kwargs_keys)):
            kwargs[kwargs_keys[idx]] = values[idx + args_len]
        self.push(f(*args, **kwargs))

    def call_function_var_op(self, arg: int) -> None:
        args = self.pop()
        f = self.pop()
        self.push(f(args, {}))

    def load_fast_op(self, name: str) -> None:
        if name in self.locals:
            val = self.locals[name]
        else:
            raise UnboundLocalError(
                "local variable '%s' referenced before assignment" % name
            )
        self.push(val)

    def store_fast_op(self, name: str) -> None:
        self.locals[name] = self.pop()

    def delete_fast_op(self, name: str) -> None:
        del self.locals[name]

    def store_global_op(self, name: str) -> None:
        self.globals[name] = self.pop()

    def delete_global_op(self, name: str) -> None:
        del self.globals[name]

    '''
    def call_function_op(self, arg):
        return self.call_function(arg, [], {})


    def call_function_var_op(self, arg):
        args = self.pop()
        return self.call_function(arg, args, {})

    def call_function_kw_op(self, arg):
        kwargs = self.pop()
        return self.call_function(arg, [], kwargs)

    def call_function_var_kw_op(self, arg):
        args, kwargs = self.popn(2)
        return self.call_function(arg, args, kwargs)

    def call_function(self, arg, args, kwargs):
        lenKw, lenPos = divmod(arg, 256)
        namedargs = {}
        for i in range(lenKw):
            key, val = self.popn(2)
            namedargs[key] = val
        namedargs.update(kwargs)
        posargs = self.popn(lenPos)
        posargs.extend(args)

        func = self.pop()
        if hasattr(func, 'im_func'):
            # Methods get self as an implicit first parameter.
            if func.im_self:
                posargs.insert(0, func.im_self)
            # The first parameter must be the correct type.
            if not isinstance(posargs[0], func.im_class):
                raise TypeError(
                    'unbound method %s() must be called with %s instance '
                    'as first argument (got %s instance instead)' % (
                        func.im_func.func_name,
                        func.im_class.__name__,
                        type(posargs[0]).__name__,
                    )
                )
            func = func.im_func
        retval = func(*posargs, **namedargs)
        self.push(retval)
    '''

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.8.5/library/dis.html#opcode-STORE_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.8/Python/ceval.c#L2280
        """
        self.locals[arg] = self.pop()

    def jump_if_true_or_pop_op(self, jump: int) -> None:
        val = self.top()
        if val:
            self.jump(jump)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, jump: int) -> None:
        val = self.top()
        if not val:
            self.jump(jump)
        else:
            self.pop()

    def dup_top_op(self, strange_magic: tp.Any) -> None:
        self.push(self.top())

    def dup_topx_op(self, count: int) -> None:
        items = self.popn(count)
        for i in [1, 2]:
            self.push(*items)

    def dup_top_two_op(self, strange_magic: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a, b, a, b)

    def rot_two_op(self, strange_magic: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(b, a)

    def rot_three_op(self, strange_magic: tp.Any) -> None:
        a, b, c = self.popn(3)
        self.push(c, a, b)

    def rot_four_op(self, strange_magic: tp.Any) -> None:
        a, b, c, d = self.popn(4)
        self.push(d, a, b, c)

    def yield_value_op(self, strange_magic: tp.Any) -> None:
        self.return_value = self.pop()

    UNARY_OPERATORS = {
        'POSITIVE': operator.pos,
        'NEGATIVE': operator.neg,
        'NOT': operator.not_,
        'CONVERT': repr,
        'INVERT': operator.invert,
    }

    def unaryOperator(self, op: str) -> None:
        x = self.pop()
        self.push(self.UNARY_OPERATORS[op](x))

    BINARY_OPERATORS = {
        'POWER': pow,
        'MULTIPLY': operator.mul,
        'DIVIDE': getattr(operator, 'div', lambda x, y: None),
        'FLOOR_DIVIDE': operator.floordiv,
        'TRUE_DIVIDE': operator.truediv,
        'MODULO': operator.mod,
        'ADD': operator.add,
        'SUBTRACT': operator.sub,
        'SUBSCR': operator.getitem,
        'LSHIFT': operator.lshift,
        'RSHIFT': operator.rshift,
        'AND': operator.and_,
        'XOR': operator.xor,
        'OR': operator.or_,
    }

    def binaryOperator(self, op: str) -> None:
        x, y = self.popn(2)
        self.push(self.BINARY_OPERATORS[op](x, y))

    def inplaceOperator(self, op: str) -> None:
        x, y = self.popn(2)
        if op == 'POWER':
            x **= y
        elif op == 'MULTIPLY':
            x *= y
        elif op in ['DIVIDE', 'FLOOR_DIVIDE']:
            x //= y
        elif op == 'TRUE_DIVIDE':
            x /= y
        elif op == 'MODULO':
            x %= y
        elif op == 'ADD':
            x += y
        elif op == 'SUBTRACT':
            x -= y
        elif op == 'LSHIFT':
            x <<= y
        elif op == 'RSHIFT':
            x >>= y
        elif op == 'AND':
            x &= y
        elif op == 'XOR':
            x ^= y
        elif op == 'OR':
            x |= y
        else:  # pragma: no cover
            raise VMError("Unknown in-place operator: %r" % op)
        self.push(x)

    COMPARE_OPERATORS = {
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '>=': operator.ge,
        'in': lambda x, y: x in y,
        'not in': lambda x, y: x not in y,
        'is': lambda x, y: x is y,
        'is not': lambda x, y: x is not y,
        'bla': lambda x, y: issubclass(x, Exception) and issubclass(x, y),
    }

    def compare_op_op(self, opnum: str) -> None:
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[opnum](x, y))

    def pop_jump_if_true_op(self, jump: int) -> None:
        val = self.pop()
        if val:
            self.jump(jump)

    def pop_jump_if_false_op(self, jump: int) -> None:
        val = self.pop()
        if not val:
            self.jump(jump)

    def raise_varargs_op(self, argc: int) -> tp.Any:
        cause = exc = None
        if argc == 2:
            cause = self.pop()
            exc = self.pop()
        elif argc == 1:
            exc = self.pop()
        return self.do_raise(exc, cause)

    def do_raise(self, exc: tp.Any, cause: tp.Any) -> str:
        if exc is None:  # reraise
            exc_type, val, tb = self.last_exception
            if exc_type is None:
                return 'exception'  # error
            else:
                return 'reraise'

        elif type(exc) == type:
            # As in `raise ValueError`
            exc_type = exc
            val = exc()  # Make an instance.
        elif isinstance(exc, BaseException):
            # As in `raise ValueError('foo')`
            exc_type = type(exc)
            val = exc
        else:
            return 'exception'  # error

        # If you reach this point, you're guaranteed that
        # val is a valid exception instance and exc_type is its class.
        # Now do a similar thing for the cause, if present.
        if cause:
            if type(cause) == type:
                cause = cause()
            elif not isinstance(cause, BaseException):
                return 'exception'  # error

            val.__cause__ = cause

        self.last_exception = exc_type, val, val.__traceback__
        return 'exception'

    def unpack_sequence_op(self, count: tp.Any) -> None:
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def extended_arg_op(self, ext: int) -> None:
        '''
        if self.arg_extended:
            self.pop()
            self.push(ext << 8)
        else:
            self.push(ext << 8)
        self.arg_extended = True
        '''
        pass

    def jump_forward_op(self, jump: int) -> None:
        self.jump(jump)

    def jump_absolute_op(self, jump: int) -> None:
        self.jump(jump)

    def get_iter_op(self, strange_magic: tp.Any) -> None:
        self.push(iter(self.pop()))

    def for_iter_op(self, jump: int) -> None:
        iterobj = self.top()
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(jump)

    def store_subscr_op(self, strange_magic: tp.Any) -> None:
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def delete_subscr_op(self, strange_magic: tp.Any) -> None:
        obj, subscr = self.popn(2)
        del obj[subscr]

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.popn(count)))

    def build_list_op(self, count: int) -> None:
        self.push(self.popn(count))

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def build_const_key_map_op(self, count: int) -> None:
        # not sure
        dictionary = {}
        keys = self.pop()
        values = self.popn(count)
        for idx in range(count):
            dictionary[keys[idx]] = values[idx]
        self.push(dictionary)

    def build_slice_op(self, count: int) -> None:
        if count == 2:
            x, y = self.popn(2)
            self.push(slice(x, y))
        elif count == 3:
            x, y, z = self.popn(3)
            self.push(slice(x, y, z))
        else:  # pragma: no cover
            raise VMError("Strange BUILD_SLICE count: %r" % count)

    def build_map_op(self, size: int) -> None:
        # size is ignored.
        self.push({})

    def store_map_op(self) -> None:
        the_map, val, key = self.popn(3)
        the_map[key] = val
        self.push(the_map)

    def list_append_op(self, count: int) -> None:
        val = self.pop()
        the_list = self.peek(count)
        the_list.append(val)

    def set_add_op(self, count: int) -> None:
        val = self.pop()
        the_set = self.peek(count)
        the_set.add(val)

    def map_add_op(self, count: int) -> None:
        val, key = self.popn(2)
        the_map = self.peek(count)
        the_map[key] = val

    def sliceOperator(self, op: tp.Any) -> None:
        start = 0
        end = None  # we will take this to mean end
        op, count = op[:-2], int(op[-1])
        if count == 1:
            start = self.pop()
        elif count == 2:
            end = self.pop()
        elif count == 3:
            end = self.pop()
            start = self.pop()
        ll = self.pop()
        if end is None:
            end = len(ll)
        if op.startswith('STORE_'):
            ll[start:end] = self.pop()
        elif op.startswith('DELETE_'):
            del ll[start:end]
        else:
            self.push(ll[start:end])

    def import_name_op(self, name: str) -> None:
        level, fromlist = self.popn(2)
        self.push(
            __import__(name, self.globals, self.locals, fromlist, level)
        )

    def import_star_op(self, strange_magic: tp.Any) -> None:
        # TODO: this doesn't use __all__ properly.
        mod = self.pop()
        for attr in dir(mod):
            if attr[0] != '_':
                self.locals[attr] = getattr(mod, attr)

    def import_from_op(self, name: str) -> None:
        mod = self.top()
        self.push(getattr(mod, name))

    def load_attr_op(self, attr: tp.Any) -> None:
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def store_attr_op(self, name: str) -> None:
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def delete_attr_op(self, name: str) -> None:
        obj = self.pop()
        delattr(obj, name)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_text_or_obj: code for interpreting
        """
        globals_context: tp.Dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
