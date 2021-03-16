import pytest
import sys
import typing as tp

# pls don't use `inspect` and `FunctionType`
from . import function_type_ban  # noqa
sys.modules['inspect'] = None  # type: ignore # noqa

from . import cases  # noqa
from . import vm_scorer  # noqa
from . import vm_runner  # noqa
from . import vm  # noqa

IDS = [test.name for test in cases.TEST_CASES]
TESTS = [test.text_code for test in cases.TEST_CASES]
SCORER = vm_scorer.Scorer(TESTS)
SCORES = [SCORER.score(test) for test in TESTS]


class Scorer:
    def __init__(self) -> None:
        self._score = 0.

    def score(self, score: float) -> None:
        self._score += score

    def __str__(self) -> str:
        return f"Full score is: {self._score}"


def test_version() -> None:
    """
    To do this task you need python=3.8.5
    """
    assert '3.8.5' == sys.version.split(' ', maxsplit=1)[0]


@pytest.fixture(scope="module")
def full_score() -> tp.Generator[Scorer, None, None]:
    scorer = Scorer()
    yield scorer
    print(scorer)


@pytest.mark.parametrize("test,score", zip(cases.TEST_CASES, SCORES), ids=IDS)
def test_all_cases(test: cases.Case, score: float, full_score: Scorer) -> None:
    """
    Compare all test cases with etalon
    :param test: test case to check
    :param score: score for test if passed
    """
    code = vm_runner.compile_code(test.text_code)
    globals_context: tp.Dict[str, tp.Any] = {}
    vm_out, vm_err, vm_exc = vm_runner.execute(code, vm.VirtualMachine().run)
    py_out, py_err, py_exc = vm_runner.execute(code, eval, globals_context, globals_context)

    assert vm_out == py_out

    if py_exc is not None:
        assert vm_exc == py_exc
    else:
        assert vm_exc is None

    # Write to stderr for subsequent parsing
    sys.stderr.write("test result score: {}\n".format(score))
    sys.stderr.flush()
    # Write score into scorer
    full_score.score(score)
