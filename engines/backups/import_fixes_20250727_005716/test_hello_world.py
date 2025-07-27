# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


def test_hello_world():  # type: ignore[no-untyped-def]
    assert "Hello, World!" == "Hello, World!"
