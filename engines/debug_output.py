#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-except,unused-variable,too-many-arguments
# pylint: disable=too-few-public-methods,invalid-name,unused-argument
# flake8: noqa: E501,F401,F821,A001,F403
# mypy: disable-error-code=no-untyped-def,misc


# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long
    from nlp.syllable.engine import SyllabicUnitEngine  # noqa: F401
    import pprint  # noqa: F401,
    def debug_output():  # type: ignore[no-untyped-def]
    """TODO: Add docstring."""
    engine = SyllabicUnitEngine()

    print("Testing syllabification output structure...")
    result = engine.syllabify_text('ؤ')

    print("Result keys:", list(result.keys()))
    print("\nFull result:")
    pprint.pprint(result, width=80, depth=3)


if __name__ == "__main__":
    debug_output()
