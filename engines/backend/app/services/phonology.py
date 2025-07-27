"""
Phonology service for processing Arabic text
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import time

# Import the phonological engine
from core.nlp.phonological.arabic_phonological_engine import PhonologicalEngine

logger = logging.getLogger(__name__)


class PhonologyService:
    """Service for phonological processing of Arabic text"""

    def __init__(self):
        """Initialize the phonology service with the phonological engine"""
        logger.info("Initializing phonology service")
        self.engine = PhonologicalEngine()

    def process_text(
        self, text: str, apply_recursively: bool = False, max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Process text through phonological rules

        Args:
            text: Text to process
            apply_recursively: Whether to apply rules recursively
            max_iterations: Maximum number of iterations for recursive application

        Returns:
            Processed text and analysis
        """
        # Store original configuration
        original_config = {
            "apply_recursively": self.engine.config.get("apply_recursively", False),
            "max_iterations": self.engine.config.get("max_iterations", 10),
        }

        try:
            # Update configuration for this request
            self.engine.config["apply_recursively"] = apply_recursively
            self.engine.config["max_iterations"] = max_iterations

            # Process the text
            start_time = time.time()
            result = self.engine.process_text(text)
            processing_time = time.time() - start_time

            # Add processing time to result
            result["processing_time"] = processing_time
            return result

        finally:
            # Restore original configuration
            self.engine.config["apply_recursively"] = original_config[
                "apply_recursively"
            ]
            self.engine.config["max_iterations"] = original_config["max_iterations"]

    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Get information about available phonological rules

        Returns:
            List of rule information dictionaries
        """
        rules_info = []

        for rule_name in self.engine.config.get("rules_order", []):
            enabled = self.engine.config.get("rules_enabled", {}).get(rule_name, False)
            rule_data = self.engine.rule_data.get(rule_name, {})

            rules_info.append(
                {
                    "name": rule_name,
                    "enabled": enabled,
                    "description": rule_data.get(
                        "description", "No description available"
                    ),
                    "examples": rule_data.get("examples", []),
                    "transformations": rule_data.get("transformations", []),
                }
            )

        return rules_info

    def apply_rule(self, text: str, rule_name: str) -> Dict[str, Any]:
        """
        Apply a specific phonological rule to text

        Args:
            text: Text to process
            rule_name: Name of the rule to apply

        Returns:
            Result of rule application

        Raises:
            ValueError: If rule is not found or not enabled
        """
        # Check if rule exists and is enabled
        if rule_name not in self.engine.config.get("rules_order", []):
            raise ValueError(f"Rule '{rule_name}' not found")

        if not self.engine.config.get("rules_enabled", {}).get(rule_name, False):
            raise ValueError(f"Rule '{rule_name}' is not enabled")

        # Find the rule instance
        rule = next((r for r in self.engine.rules if r.rule_name == rule_name), None)
        if not rule:
            raise ValueError(f"Rule '{rule_name}' implementation not found")

        # Apply the rule to each word in the text
        words = text.split()
        results = []

        for word in words:
            phonemes = list(word)  # Simple tokenization
            before = phonemes.copy()
            after = rule.apply(phonemes)

            results.append(
                {
                    "word": word,
                    "result": "".join(after),
                    "changed": before != after,
                    "transformations": rule.get_transformations(),
                }
            )

        return {
            "original_text": text,
            "processed_text": " ".join(r["result"] for r in results),
            "rule_name": rule_name,
            "word_results": results,
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the phonological engine

        Returns:
            Status information including rules, statistics, and configuration
        """
        return {
            "status": "active",
            "enabled_rules": self.engine.get_enabled_rules(),
            "statistics": self.engine.statistics,
            "config": self.engine.config,
            "validation": self.engine.validate_configuration(),
        }

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the phonological engine configuration

        Args:
            config: Configuration dictionary to update

        Returns:
            Updated status information
        """
        old_config = self.engine.config.copy()
        self.engine.config.update(config)

        # Reinitialize rules if rules configuration changed
        if (
            "rules_order" in config
            or "rules_enabled" in config
            or any(key.startswith("rule_") for key in config)
        ):
            self.engine._init_rules()

        return {
            "status": "success",
            "old_config": old_config,
            "new_config": self.engine.config,
            "validation": self.engine.validate_configuration(),
        }

    def reset_statistics(self) -> Dict[str, Any]:
        """
        Reset all processing statistics

        Returns:
            Reset confirmation and current statistics
        """
        self.engine.reset_statistics()
        return {
            "status": "success",
            "message": "Statistics reset successfully",
            "statistics": self.engine.statistics,
        }
