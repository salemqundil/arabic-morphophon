"""
Soft-Logic Rules Engine for Arabic Morphophonological Analysis
محرك القواعد المنطقية الناعمة للتحليل الصرفي الصوتي العربي

Implements soft-logic constraints and rules from the hierarchical architecture:
- Phonological rules (assimilation, elision, etc.)
- Morphological rules (pattern constraints)
- Syntactic rules (agreement, government)
- Semantic rules (selectional restrictions)
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data math
import_data re
from typing import_data Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import_data torch
    import_data torch.nn as nn
    import_data torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore

# Rule types and priorities
RULE_TYPES = {
    'phonological': 1,     # Highest priority - sound changes
    'morphological': 2,    # Form and pattern constraints
    'syntactic': 3,        # Agreement and government
    'semantic': 4,         # Meaning-based constraints
    'stylistic': 5         # Lowest priority - style rules
}

# Arabic phonological processes
PHONOLOGICAL_PROCESSES = {
    'assimilation': {
        'total': ['نْ + ب → مْب', 'نْ + م → مّ'],
        'partial': ['نْ + ك → نْك'],
        'regressive': ['ت + د → دّ'],
        'progressive': ['س + ت → سْت']
    },
    'elision': {
        'vowel_deletion': ['ا + ا → ا', 'ي + ي → ي'],
        'consonant_deletion': ['تْ + ت → ت']
    },
    'epenthesis': {
        'vowel_insertion': ['CCC → CVC'],
        'consonant_insertion': ['VV → VyV']
    },
    'metathesis': {
        'consonant_swap': ['فعل → لعف (rare)']
    }
}

# Morphological constraints
MORPHOLOGICAL_CONSTRAINTS = {
    'root_harmony': {
        'no_identical_radicals': 'R1 ≠ R2 ≠ R3 (generally)',
        'guttural_restrictions': 'ح،ع،غ،خ restrictions in certain forms'
    },
    'pattern_constraints': {
        'form_iv_hamza': 'Form IV must begin with hamza',
        'gemination_rules': 'C2 = C3 in certain patterns'
    },
    'vocalization': {
        'short_vowel_restrictions': 'Context-dependent vowel harmony',
        'long_vowel_constraints': 'Position-sensitive lengthening'
    }
}

# Syntactic rules
SYNTACTIC_RULES = {
    'agreement': {
        'gender': 'Adjective agrees with noun in gender',
        'number': 'Verb agrees with subject in number',
        'case': 'Adjective agrees with noun in case'
    },
    'government': {
        'transitive_object': 'Transitive verb requires object',
        'preposition_case': 'Preposition governs genitive case',
        'idafa_case': 'Second noun in idafa is genitive'
    }
}

class RuleViolation:
    """
    Represents a violation of a soft-logic rule
    يمثل انتهاك قاعدة منطقية ناعمة
    """
    
    def __init__(self, rule_name: str, rule_type: str, severity: float,
                 description: str, context: Dict, suggested_fix: Optional[str] = None):
        self.rule_name = rule_name
        self.rule_type = rule_type
        self.severity = severity  # 0.0 to 1.0
        self.description = description
        self.context = context
        self.suggested_fix = suggested_fix
        self.confidence = 1.0
        
    def to_dict(self) -> Dict:
        """Convert violation to dictionary"""
        return {
            'rule_name': self.rule_name,
            'rule_type': self.rule_type,
            'severity': self.severity,
            'description': self.description,
            'context': self.context,
            'suggested_fix': self.suggested_fix,
            'confidence': self.confidence
        }

class SoftLogicRule:
    """
    Base class for soft-logic rules
    فئة أساسية للقواعد المنطقية الناعمة
    """
    
    def __init__(self, name: str, rule_type: str, priority: int = 1, 
                 margin: float = 0.5, enabled: bool = True):
        self.name = name
        self.rule_type = rule_type
        self.priority = priority
        self.margin = margin  # Soft margin for violations
        self.enabled = enabled
        self.violation_count = 0
        self.application_count = 0
        
    def apply(self, text: str, analysis: Dict, metadata: Dict, 
              embeddings: Optional[List[List[float]]] = None) -> Optional[RuleViolation]:
        """
        Apply rule to text and analysis
        
        Args:
            text: Input Arabic text
            analysis: Analysis results from engine
            metadata: Additional metadata
            embeddings: Optional node embeddings
            
        Returns:
            RuleViolation if rule is violated, None otherwise
        """
        if not self.enabled:
            return None
            
        self.application_count += 1
        violation = self._check_rule(text, analysis, metadata, embeddings)
        
        if violation:
            self.violation_count += 1
            
        return violation
        
    def _check_rule(self, text: str, analysis: Dict, metadata: Dict,
                   embeddings: Optional[List[List[float]]] = None) -> Optional[RuleViolation]:
        """Override in subclasses to implement specific rule logic"""
        raise NotImplementedError("Subclasses must implement _check_rule")
        
    def get_statistics(self) -> Dict:
        """Get rule application statistics"""
        violation_rate = (
            self.violation_count / self.application_count 
            if self.application_count > 0 else 0.0
        )
        
        return {
            'name': self.name,
            'type': self.rule_type,
            'applications': self.application_count,
            'violations': self.violation_count,
            'violation_rate': violation_rate,
            'enabled': self.enabled
        }

class PhonologicalAssimilationRule(SoftLogicRule):
    """
    Rule for checking phonological assimilation patterns
    قاعدة فحص أنماط الإدغام الصوتي
    """
    
    def __init__(self):
        super().__init__(
            name="phonological_assimilation",
            rule_type="phonological",
            priority=1,
            margin=0.3
        )
        
        # Assimilation patterns that should occur
        self.expected_assimilations = {
            'نب': 'مب',  # nun + ba -> mim + ba
            'نم': 'مم',  # nun + mim -> geminated mim
            'تد': 'دد',  # ta + dal -> geminated dal
            'دت': 'تت'   # dal + ta -> geminated ta
        }
        
    def _check_rule(self, text: str, analysis: Dict, metadata: Dict,
                   embeddings: Optional[List[List[float]]] = None) -> Optional[RuleViolation]:
        """Check for missing assimilation"""
        
        for pattern, expected in self.expected_assimilations.items():
            if pattern in text and expected not in text:
                # Calculate severity based on frequency
                pattern_count = text.count(pattern)
                severity = min(pattern_count * 0.2, 1.0)
                
                return RuleViolation(
                    rule_name=self.name,
                    rule_type=self.rule_type,
                    severity=severity,
                    description=f"Expected assimilation: {pattern} → {expected}",
                    context={'pattern': pattern, 'expected': expected, 'count': pattern_count},
                    suggested_fix=f"Replace '{pattern}' with '{expected}'"
                )
                
        return None

class MorphologicalFormRule(SoftLogicRule):
    """
    Rule for checking morphological form constraints
    قاعدة فحص قيود الأوزان الصرفية
    """
    
    def __init__(self):
        super().__init__(
            name="form_iv_hamza",
            rule_type="morphological", 
            priority=2,
            margin=0.4
        )
        
    def _check_rule(self, text: str, analysis: Dict, metadata: Dict,
                   embeddings: Optional[List[List[float]]] = None) -> Optional[RuleViolation]:
        """Check if Form IV verbs begin with hamza"""
        
        # Look for Form IV indicators in analysis
        if 'detected_patterns' in analysis:
            for pattern in analysis['detected_patterns']:
                if pattern.get('form') == 'IV' or 'أفعل' in pattern.get('pattern', ''):
                    word = pattern.get('word', '')
                    if word and not word.beginswith('أ'):
                        return RuleViolation(
                            rule_name=self.name,
                            rule_type=self.rule_type,
                            severity=0.8,
                            description="Form IV verb must begin with hamza",
                            context={'word': word, 'pattern': pattern},
                            suggested_fix=f"Add hamza at beginning: أ{word}"
                        )
                        
        return None

class SyntacticAgreementRule(SoftLogicRule):
    """
    Rule for checking syntactic agreement
    قاعدة فحص التطابق النحوي
    """
    
    def __init__(self):
        super().__init__(
            name="gender_agreement",
            rule_type="syntactic",
            priority=3,
            margin=0.4
        )
        
    def _check_rule(self, text: str, analysis: Dict, metadata: Dict,
                   embeddings: Optional[List[List[float]]] = None) -> Optional[RuleViolation]:
        """Check gender agreement between nouns and adjectives"""
        
        # Simple pattern matching for demonstration
        # In practice, this would use sophisticated parsing
        
        # Look for common disagreement patterns
        disagreement_patterns = [
            (r'كتاب\s+جميلة', 'Masculine noun with feminine adjective'),
            (r'مدرسة\s+جميل', 'Feminine noun with masculine adjective')
        ]
        
        for pattern, description in disagreement_patterns:
            if re.search(pattern, text):
                return RuleViolation(
                    rule_name=self.name,
                    rule_type=self.rule_type,
                    severity=0.6,
                    description=description,
                    context={'pattern': pattern, 'text': text},
                    suggested_fix="Ensure gender agreement between noun and adjective"
                )
                
        return None

class TransitivityRule(SoftLogicRule):
    """
    Rule for checking verb transitivity constraints
    قاعدة فحص قيود تعدية الأفعال
    """
    
    def __init__(self):
        super().__init__(
            name="transitive_object",
            rule_type="syntactic",
            priority=3,
            margin=0.4
        )
        
        # Common transitive verbs that require objects
        self.transitive_verbs = {
            'كتب': 'wrote',
            'قرأ': 'read', 
            'درس': 'studied',
            'فهم': 'understood',
            'حفظ': 'memorized'
        }
        
    def _check_rule(self, text: str, analysis: Dict, metadata: Dict,
                   embeddings: Optional[List[List[float]]] = None) -> Optional[RuleViolation]:
        """Check if transitive verbs have objects"""
        
        words = text.split()
        
        for i, word in enumerate(words):
            # Remove diacritics for checking
            clean_word = re.sub(r'[ًَُِْ ٌٍ]', '', word)
            
            if clean_word in self.transitive_verbs:
                # Check if there's a potential object after the verb
                has_object = False
                
                # Simple heuristic: look for noun after verb
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    # If next word is not a verb or particle, assume it's an object
                    if not self._is_function_word(next_word):
                        has_object = True
                        
                if not has_object:
                    return RuleViolation(
                        rule_name=self.name,
                        rule_type=self.rule_type,
                        severity=0.7,
                        description=f"Transitive verb '{word}' may be missing object",
                        context={'verb': word, 'position': i, 'sentence': text},
                        suggested_fix="Add appropriate direct object after transitive verb"
                    )
                    
        return None
        
    def _is_function_word(self, word: str) -> bool:
        """Check if word is a function word (particle, pronoun, etc.)"""
        function_words = {'في', 'على', 'من', 'إلى', 'مع', 'هو', 'هي', 'أن', 'لا', 'ما'}
        clean_word = re.sub(r'[ًَُِْ ٌٍ]', '', word)
        return clean_word in function_words

class AdvancedRulesEngine:
    """
    Advanced Rules Engine for Arabic Morphophonological Analysis
    محرك القواعد المتقدم للتحليل الصرفي الصوتي العربي
    
    Orchestrates multiple soft-logic rules with priority-based application
    and violation detection with suggested repairs.
    """
    
    def __init__(self, enable_neural: bool = True):
        """
        Initialize rules engine
        
        Args:
            enable_neural: Whether to use neural components
        """
        self.enable_neural = enable_neural and TORCH_AVAILABLE
        self.rules: List[SoftLogicRule] = []
        self.global_stats = {
            'total_applications': 0,
            'total_violations': 0,
            'rules_applied': 0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default set of rules"""
        self.rules = [
            PhonologicalAssimilationRule(),
            MorphologicalFormRule(),
            SyntacticAgreementRule(),
            TransitivityRule()
        ]
        
        # Sort by priority (lower number = higher priority)
        self.rules.sort(key=lambda r: r.priority)
        
    def add_rule(self, rule: SoftLogicRule):
        """Add a new rule to the engine"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
        
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True
        return False
        
    def enable_rule(self, rule_name: str, enabled: bool = True) -> bool:
        """Enable or disable a specific rule"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                return True
        return False
        
    def validate_text(self, text: str, analysis: Dict, metadata: Optional[Dict] = None,
                     embeddings: Optional[List[List[float]]] = None) -> List[RuleViolation]:
        """
        Validate text against all enabled rules
        
        Args:
            text: Input Arabic text
            analysis: Analysis results from morphophonological engine
            metadata: Additional metadata
            embeddings: Optional node embeddings
            
        Returns:
            List of rule violations
        """
        if metadata is None:
            metadata = {}
            
        violations = []
        
        for rule in self.rules:
            if rule.enabled:
                violation = rule.apply(text, analysis, metadata, embeddings)
                if violation:
                    violations.append(violation)
                    
                self.global_stats['total_applications'] += 1
                if violation:
                    self.global_stats['total_violations'] += 1
                    
        self.global_stats['rules_applied'] = len([r for r in self.rules if r.enabled])
        
        return violations
        
    def get_rule_statistics(self) -> List[Dict]:
        """Get statistics for all rules"""
        return [rule.get_statistics() for rule in self.rules]
        
    def get_global_statistics(self) -> Dict:
        """Get global engine statistics"""
        violation_rate = (
            self.global_stats['total_violations'] / self.global_stats['total_applications']
            if self.global_stats['total_applications'] > 0 else 0.0
        )
        
        return {
            **self.global_stats,
            'violation_rate': violation_rate,
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled])
        }
        
    def hard_repair(self, text: str, violations: List[RuleViolation]) -> Tuple[str, List[str]]:
        """
        Attempt to repair violations using hard constraints
        
        Args:
            text: Original text
            violations: List of violations to repair
            
        Returns:
            Tuple of (repaired_text, list_of_changes_made)
        """
        repaired_text = text
        changes_made = []
        
        # Sort violations by severity (highest first)
        sorted_violations = sorted(violations, key=lambda v: v.severity, reverse=True)
        
        for violation in sorted_violations:
            if violation.suggested_fix:
                change = None
                # Apply suggested fix
                if violation.rule_type == 'phonological':
                    repaired_text, change = self._apply_phonological_fix(
                        repaired_text, violation
                    )
                elif violation.rule_type == 'morphological':
                    repaired_text, change = self._apply_morphological_fix(
                        repaired_text, violation
                    )
                elif violation.rule_type == 'syntactic':
                    repaired_text, change = self._apply_syntactic_fix(
                        repaired_text, violation
                    )
                    
                if change:
                    changes_made.append(change)
                    
        return repaired_text, changes_made
        
    def _apply_phonological_fix(self, text: str, violation: RuleViolation) -> Tuple[str, Optional[str]]:
        """Apply phonological repair"""
        if 'pattern' in violation.context and 'expected' in violation.context:
            pattern = violation.context['pattern']
            expected = violation.context['expected']
            
            if pattern in text:
                new_text = text.replace(pattern, expected)
                change = f"Phonological: {pattern} → {expected}"
                return new_text, change
                
        return text, None
        
    def _apply_morphological_fix(self, text: str, violation: RuleViolation) -> Tuple[str, Optional[str]]:
        """Apply morphological repair"""
        if violation.rule_name == 'form_iv_hamza' and 'word' in violation.context:
            word = violation.context['word']
            if word in text and not word.beginswith('أ'):
                new_word = 'أ' + word
                new_text = text.replace(word, new_word)
                change = f"Morphological: {word} → {new_word}"
                return new_text, change
                
        return text, None
        
    def _apply_syntactic_fix(self, text: str, violation: RuleViolation) -> Tuple[str, Optional[str]]:
        """Apply syntactic repair"""
        # Syntactic repairs are often complex and context-dependent
        # For now, just log the suggested fix
        change = f"Syntactic suggestion: {violation.suggested_fix}"
        return text, change
        
    def get_info(self) -> Dict:
        """Get information about the rules engine"""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'rule_types': list(RULE_TYPES.keys()),
            'neural_enabled': self.enable_neural,
            'torch_available': TORCH_AVAILABLE,
            'phonological_processes': list(PHONOLOGICAL_PROCESSES.keys()),
            'morphological_constraints': list(MORPHOLOGICAL_CONSTRAINTS.keys()),
            'syntactic_rules': list(SYNTACTIC_RULES.keys()),
            'global_stats': self.global_stats
        }

# Example usage and testing
if __name__ == "__main__":
    print("⚖️ Testing Arabic Soft-Logic Rules Engine")
    print("=" * 50)
    
    # Initialize rules engine
    engine = AdvancedRulesEngine(enable_neural=TORCH_AVAILABLE)
    
    # Test text with potential violations
    test_text = "كتب الطالب"  # "The student wrote" (missing object)
    
    # Mock analysis result
    analysis = {
        'identified_roots': [{'root': 'كتب', 'confidence': 0.9}],
        'detected_patterns': []
    }
    
    print(f"📝 Test text: {test_text}")
    
    # Validate text
    violations = engine.validate_text(test_text, analysis)
    
    print(f"🚨 Violations found: {len(violations)}")
    for i, violation in enumerate(violations, 1):
        print(f"  {i}. {violation.rule_name}: {violation.description}")
        print(f"     Severity: {violation.severity:.2f}")
        if violation.suggested_fix:
            print(f"     Suggestion: {violation.suggested_fix}")
            
    # Test hard repair
    if violations:
        print(f"\n🔧 Testing hard repair...")
        repaired_text, changes = engine.hard_repair(test_text, violations)
        print(f"Original: {test_text}")
        print(f"Repaired: {repaired_text}")
        print(f"Changes: {changes}")
        
    # Show statistics
    print(f"\n📊 Engine statistics:")
    stats = engine.get_global_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print(f"\n📋 Rule statistics:")
    rule_stats = engine.get_rule_statistics()
    for stat in rule_stats:
        print(f"  {stat['name']}: {stat['applications']} apps, {stat['violations']} violations")
