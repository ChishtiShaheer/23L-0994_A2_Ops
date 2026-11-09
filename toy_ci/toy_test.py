"""
Toy CI Test File for Task 6.6
Toggle FAIL_TEST between True and False to demonstrate Red and Green CI runs.
"""

FAIL_TEST = False  # Set to True to trigger RED run, then False for GREEN run

def test_math_logic():
    a = 10
    b = 20
    assert a + b == 30

def test_deliberate_failure():
    if FAIL_TEST:
        # Deliberate failure for Task 6.6 Evidence
        assert False, "Deliberate test failure for Task 6.6 CI Red run screenshot!"
    else:
        assert True
