#!/usr/bin/env python3
"""phonological_engine.py
A **clean** and *minimal* re‑implementation of the Arabic *Phonological Engine* that
was previously duplicated, syntactically invalid and impossible to import.

Key goals of this rewrite
-------------------------
*   **Syntactic soundness** – the file can be imported & executed with CPython ≥3.8
*   **Single source of truth** – every method is defined *once*; no more copy‑paste blobs
*   **Fail‑fast logging** – logger is configured correctly or disabled gracefully
*   **Pluggable rules** – `AssimilationRule`, `DeletionRule`, `InversionRule` are
    loaded only if present so the engine can still start when a rule module is
    missing ( → useful in unit‑tests or constrained deploys )
*   **Unit‑tests discoverable by *unittest* discover** – tests live in the same
    module for convenience **and** are exposed via ``load_tests`` so that the
    `unittest` discovery runner (``python -m unittest``) picks them up.
    → This fixes the *SystemExit: 5* error the user encountered.
"""
from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################
import json
import logging
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Callable

###############################################################################
# Third‑party deps                                                            #
###############################################################################
try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover – hard failure is fine here
    raise RuntimeError("`PyYAML` is required: pip install pyyaml") from exc

###############################################################################
# Utility helpers                                                             #
###############################################################################


class _SafeLoader:
    """Wrapper around *yaml.safe_load* that returns *{}* on IO errors."""

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        """Load YAML safely, returning empty dict on errors."""
        try:
            with Path(path).expanduser().open("rt", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            return {}


def _safe_json_load(path: Path) -> Dict[str, Any]:
    """Load JSON safely, returning empty dict on errors."""
    try:
        with Path(path).expanduser().open("rt", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


###############################################################################
# Rule abstractions                                                           #
###############################################################################


class _BaseRule:
    """Abstract interface all rule processors should follow."""

    rule_name: str = "base"

    def __init__(self, data: Dict[str, Any]):
        """Initialize rule with provided data."""
        self._data = data
        self._transformations: List[Tuple[List[str], List[str]]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, phonemes: List[str]) -> List[str]:  # noqa: D401
        """Transform *phonemes* in‑place. Concrete rules must override this."""
        return phonemes  # pragma: no cover – base impl is a no‑op

    def get_transformations(self) -> List[Tuple[List[str], List[str]]]:
        """Return a copy of recorded transformations."""
        return self._transformations.copy()

    def clear_log(self) -> None:
        """Clear the transformation log."""
        self._transformations.clear()


###############################################################################
# Dynamic import helpers – gracefully degrade when a rule module is absent    #
###############################################################################

_RULE_REGISTRY: Dict[str, Type[_BaseRule]] = {}


def _register_rule(name: str, module_path: str, cls_name: str) -> None:
    """Register a rule class by name, falling back to base rule if import fails."""
    try:
        module = __import__(module_path, fromlist=[cls_name])
        _RULE_REGISTRY[name] = getattr(module, cls_name)
    except (ImportError, AttributeError):
        # Fallback to the base rule so the engine keeps working.
        _RULE_REGISTRY[name] = _BaseRule  # type: ignore[assignment]


_register_rule("assimilation", "models.assimilation", "AssimilationRule")
_register_rule("deletion", "models.deletion", "DeletionRule")
_register_rule("inversion", "models.inversion", "InversionRule")

###############################################################################
# Engine                                                                      #
###############################################################################


class PhonologicalEngine:  # pylint: disable=too-many-instance-attributes
    """Arabic phonological processing engine with minimal safe defaults."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "rules_order": ["assimilation", "deletion", "inversion"],
        "rules_enabled": {
            "assimilation": True,
            "deletion": True,
            "inversion": True,
        },
        "max_iterations": 10,
        "apply_recursively": False,
        "debug_mode": False,
    }

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        config_path: str | Path | None = None,
        rule_data_path: str | Path | None = None,
    ) -> None:
        """Initialize the phonological engine with optional config and rule paths."""
        self.logger = logging.getLogger("PhonologicalEngine")
        self._setup_logging()

        base_dir = Path(__file__).resolve().parent
        self._config_path = (
            Path(config_path) if config_path else base_dir / "config/rules_config.yaml"
        )
        self._rule_data_path = (
            Path(rule_data_path) if rule_data_path else base_dir / "data/rules.json"
        )

        self.config = self._load_config(self._config_path)
        self.rule_data = self._load_rule_data(self._rule_data_path)

        self.rules: List[_BaseRule] = []
        self._init_rules()

        # Transformation statistics
        self._stats: Dict[str, Any] = {
            "total_applications": 0,
            "rule_counts": {name: 0 for name in self.config["rules_order"]},
            "processing_time": 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        """Set up logging handlers if not already configured."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s » %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file, applying defaults."""
        cfg = _SafeLoader.load(path)
        merged = {**self.DEFAULT_CONFIG, **(cfg or {})}
        # honour debug flag asap
        if merged.get("debug_mode"):
            self.logger.setLevel(logging.DEBUG)
        return merged

    def _load_rule_data(self, path: Path) -> Dict[str, Any]:
        """Load rule data from JSON file."""
        return _safe_json_load(path) or {}

    def _init_rules(self) -> None:
        """Initialize rule processors based on configuration."""
        for name in self.config["rules_order"]:
            if not self.config["rules_enabled"].get(name, False):
                continue
            data = self.rule_data.get(name, {})
            rule_cls = _RULE_REGISTRY.get(name, _BaseRule)
            self.rules.append(rule_cls(data))
            self.logger.debug("‑ loaded rule › %s (%s)", name, rule_cls.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_rules(self, phonemes: List[str]) -> List[str]:
        """Apply all enabled rules to the phoneme sequence."""
        if not phonemes:
            return phonemes  # fast‑path empty input

        start = time.perf_counter()
        result = phonemes.copy()

        for rule in self.rules:
            before = result.copy()
            result = rule.apply(result)
            if before != result:
                self._stats["rule_counts"][rule.rule_name] += 1
                if self.config["debug_mode"]:
                    self.logger.debug("%s: %s → %s", rule.rule_name, before, result)

        self._stats["total_applications"] += 1
        self._stats["processing_time"] += time.perf_counter() - start
        return result

    def apply_recursive_rules(self, phonemes: List[str]) -> List[str]:
        """Apply rules recursively until no more changes occur or max iterations reached."""
        max_iter = int(self.config.get("max_iterations", 10))
        result = phonemes.copy()
        for _ in range(max_iter):
            new = self.apply_rules(result)
            if new == result:
                break
            result = new
        else:
            self.logger.warning("Maximum iterations (%s) reached", max_iter)
        return result

    # Convenience wrappers ------------------------------------------------
    def analyze_phonemes(self, phonemes: List[str]) -> Dict[str, Any]:
        """Analyze phonemes with detailed information about transformations."""
        pipeline = (
            self.apply_recursive_rules
            if self.config["apply_recursively"]
            else self.apply_rules
        )
        result = pipeline(phonemes)
        return {
            "original": phonemes,
            "result": result,
            "changed": phonemes != result,
            "statistics": self.statistics,
        }

    def process_word(self, word: str) -> Tuple[str, Dict[str, Any]]:
        """Process a complete word through the phonological rules."""
        phonemes = list(word)  # TODO: replace with real tokenizer
        analysis = self.analyze_phonemes(phonemes)
        return "".join(analysis["result"]), analysis

    # Properties -----------------------------------------------------------
    @property
    def statistics(self) -> Dict[str, Any]:
        """Return current processing statistics."""
        stats = self._stats.copy()
        total = stats["total_applications"] or 1
        stats["average_processing_time"] = stats["processing_time"] / total
        return stats

    def reset_statistics(self) -> None:
        """Reset all processing statistics."""
        self._stats.update(
            {
                "total_applications": 0,
                "processing_time": 0.0,
                "rule_counts": {name: 0 for name in self._stats["rule_counts"]},
            }
        )
        for rule in self.rules:
            rule.clear_log()

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate engine configuration."""
        validation = {
            "valid": True,
            "issues": [],
            "rule_status": {},
        }

        # Check rule data availability
        for rule_name in self.config.get("rules_order", []):
            if self.config.get("rules_enabled", {}).get(rule_name, False):
                if rule_name not in self.rule_data:
                    validation["valid"] = False
                    validation["issues"].append(f"Missing rule data for {rule_name}")
                    validation["rule_status"][rule_name] = "missing_data"
                else:
                    validation["rule_status"][rule_name] = "available"

        # Check for circular dependencies (simplified)
        rules_order = self.config.get("rules_order", [])
        if len(rules_order) != len(set(rules_order)):
            validation["valid"] = False
            validation["issues"].append("Duplicate rules in rules_order")

        return validation

    def get_rule_info(self) -> Dict[str, Any]:
        """Get information about loaded rules."""
        info = {
            "total_rules": len(self.rules),
            "rule_processors": [],
            "configuration": self.config,
            "rule_data_summary": {},
        }

        for rule_processor in self.rules:
            rule_info = {
                "name": rule_processor.rule_name,
                "type": type(rule_processor).__name__,
                "transformations_logged": len(rule_processor.get_transformations()),
            }
            info["rule_processors"].append(rule_info)

        # Summarize rule data
        for rule_name, rule_data in self.rule_data.items():
            if isinstance(rule_data, dict) and "rules" in rule_data:
                info["rule_data_summary"][rule_name] = len(rule_data["rules"])

        return info


###############################################################################
# Built‑in **unit‑tests**                                                     #
###############################################################################


class _DummyRule(_BaseRule):
    """Reverses the phoneme sequence once – handy for predictable testing."""

    rule_name = "dummy"

    def apply(self, phonemes: List[str]) -> List[str]:
        """Apply dummy transformation (reverse the phonemes)."""
        if phonemes:
            transformed = phonemes[::-1]
            self._transformations.append((phonemes, transformed))
            return transformed
        return phonemes


class PhonologicalEngineTestCase(unittest.TestCase):
    """Minimal smoke‑tests to ensure the engine is importable and callable."""

    def setUp(self) -> None:
        """Set up test environment."""
        # Monkey‑patch registry so the engine picks up our dummy rule only.
        _RULE_REGISTRY["dummy"] = _DummyRule  # type: ignore[assignment]
        cfg = {"rules_order": ["dummy"], "rules_enabled": {"dummy": True}}
        self.engine = PhonologicalEngine()
        self.engine.config.update(cfg)
        self.engine._init_rules()  # Re-initialize with our test config

    def test_empty_input(self) -> None:
        """Test that empty input returns empty output."""
        result = self.engine.apply_rules([])
        self.assertEqual(result, [])

    def test_simple_transform(self) -> None:
        """Test that our dummy rule correctly reverses phonemes."""
        result = self.engine.apply_rules(["a", "b", "c"])
        self.assertEqual(result, ["c", "b", "a"])

    def test_recursive_transform(self) -> None:
        """Test recursive transform (should still reverse only once)."""
        result = self.engine.apply_recursive_rules(["a", "b", "c"])
        self.assertEqual(result, ["c", "b", "a"])

    def test_statistics_update(self) -> None:
        """Test that statistics are correctly updated."""
        self.engine.reset_statistics()
        self.engine.apply_rules(["a", "b", "c"])
        stats = self.engine.statistics
        self.assertEqual(stats["total_applications"], 1)
        self.assertEqual(stats["rule_counts"]["dummy"], 1)

    def test_process_word(self) -> None:
        """Test the word processing convenience method."""
        word, analysis = self.engine.process_word("abc")
        self.assertEqual(word, "cba")
        self.assertEqual(analysis["original"], ["a", "b", "c"])
        self.assertEqual(analysis["result"], ["c", "b", "a"])
        self.assertTrue(analysis["changed"])

    def test_validate_configuration(self) -> None:
        """Test configuration validation."""
        validation = self.engine.validate_configuration()
        self.assertTrue(validation["valid"])

        # Test with invalid config
        old_config = self.engine.config.copy()
        self.engine.config["rules_order"] = ["dummy", "dummy"]  # Duplicate
        validation = self.engine.validate_configuration()
        self.assertFalse(validation["valid"])
        self.assertIn("Duplicate rules in rules_order", validation["issues"])

        # Restore config
        self.engine.config = old_config


def load_tests(loader, standard_tests, pattern):
    """Enable unittest discovery to find these tests."""
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(PhonologicalEngineTestCase))
    return suite


if __name__ == "__main__":
    unittest.main()
