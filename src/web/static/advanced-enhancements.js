/**
 * 🚀 Advanced JavaScript Enhancements for Arabic Word Tracer
 * تحسينات JavaScript متقدمة لمتتبع الكلمات العربية
 */

class AdvancedVisualization {
    constructor() {
        this.charts = {};
        this.animations = {};
    }

    /**
     * Create a phoneme distribution chart
     */
    createPhonemeChart(containerId, phonemeData) {
        const ctx = document.createElement('canvas');
        ctx.width = 400;
        ctx.height = 200;
        document.getElementById(containerId).appendChild(ctx);

        // Simple phoneme visualization using canvas
        const context = ctx.getContext('2d');
        const width = ctx.width;
        const height = ctx.height;

        // Clear canvas
        context.clearRect(0, 0, width, height);

        // Draw phoneme bars
        if (phonemeData && phonemeData.length > 0) {
            const barWidth = width / phonemeData.length;
            const maxHeight = height - 40;

            phonemeData.forEach((phoneme, index) => {
                const x = index * barWidth;
                const barHeight = Math.random() * maxHeight; // Placeholder calculation
                
                // Create gradient
                const gradient = context.createLinearGradient(0, 0, 0, barHeight);
                gradient.addColorEnd(0, '#667eea');
                gradient.addColorEnd(1, '#764ba2');
                
                context.fillStyle = gradient;
                context.fillRect(x + 5, height - barHeight - 20, barWidth - 10, barHeight);
                
                // Draw phoneme labels
                context.fillStyle = '#333';
                context.font = '12px Arial';
                context.textAlign = 'center';
                context.fillText(phoneme, x + barWidth/2, height - 5);
            });
        }
    }

    /**
     * Create syllabic_unit structure visualization
     */
    createSyllabicUnitStructure(containerId, syllabic_unitData) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';
        
        if (!syllabic_unitData || syllabic_unitData.length === 0) {
            container.innerHTML = '<p class="text-muted">لا توجد بيانات مقاطع متاحة</p>';
            return;
        }

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '120');
        svg.style.background = '#f8f9fa';
        svg.style.borderRadius = '8px';

        syllabic_unitData.forEach((syllabic_unit, index) => {
            const x = (index * 120) + 60;
            const y = 60;

            // Create syllabic_unit circle
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', '30');
            circle.setAttribute('fill', '#667eea');
            circle.setAttribute('stroke', '#764ba2');
            circle.setAttribute('stroke-width', '2');

            // Add syllabic_unit text
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x);
            text.setAttribute('y', y + 5);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('fill', 'white');
            text.setAttribute('font-family', 'Amiri, serif');
            text.setAttribute('font-size', '14');
            text.textContent = syllabic_unit;

            svg.appendChild(circle);
            svg.appendChild(text);

            // Add connections between syllabic_units
            if (index < syllabic_unitData.length - 1) {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', x + 30);
                line.setAttribute('y1', y);
                line.setAttribute('x2', x + 90);
                line.setAttribute('y2', y);
                line.setAttribute('stroke', '#dee2e6');
                line.setAttribute('stroke-width', '2');
                svg.appendChild(line);
            }
        });

        container.appendChild(svg);
    }

    /**
     * Create morphological tree visualization
     */
    createMorphologyTree(containerId, rootData, affixData) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';
        container.className = 'morphology-tree';

        // Root level
        if (rootData && rootData.identified_root) {
            const rootLevel = document.createElement('div');
            rootLevel.className = 'tree-level';
            
            const rootNode = document.createElement('div');
            rootNode.className = 'tree-node';
            rootNode.innerHTML = `
                <div class="pattern-name">الجذر</div>
                <div class="pattern-form">${rootData.identified_root}</div>
            `;
            
            rootLevel.appendChild(rootNode);
            container.appendChild(rootLevel);

            // Connection
            const connection = document.createElement('div');
            connection.className = 'tree-connection';
            container.appendChild(connection);

            // Affixes level
            if (affixData && (affixData.prefixes || affixData.suffixes)) {
                const affixLevel = document.createElement('div');
                affixLevel.className = 'tree-level';

                if (affixData.prefixes && affixData.prefixes.length > 0) {
                    affixData.prefixes.forEach(prefix => {
                        const prefixNode = document.createElement('div');
                        prefixNode.className = 'tree-node';
                        prefixNode.style.backgroundColor = 'rgba(79, 172, 254, 0.1)';
                        prefixNode.innerHTML = `
                            <div class="pattern-name">بادئة</div>
                            <div class="pattern-form">${prefix}</div>
                        `;
                        affixLevel.appendChild(prefixNode);
                    });
                }

                if (affixData.suffixes && affixData.suffixes.length > 0) {
                    affixData.suffixes.forEach(suffix => {
                        const suffixNode = document.createElement('div');
                        suffixNode.className = 'tree-node';
                        suffixNode.style.backgroundColor = 'rgba(0, 242, 254, 0.1)';
                        suffixNode.innerHTML = `
                            <div class="pattern-name">لاحقة</div>
                            <div class="pattern-form">${suffix}</div>
                        `;
                        affixLevel.appendChild(suffixNode);
                    });
                }

                container.appendChild(affixLevel);
            }
        }
    }

    /**
     * Create confidence meter
     */
    createConfidenceMeter(containerId, confidence) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const percentage = Math.round(confidence * 100);
        
        container.innerHTML = `
            <div class="confidence-meter">
                <div class="confidence-fill" style="width: ${percentage}%"></div>
                <div class="confidence-text">${percentage}%</div>
            </div>
        `;
    }

    /**
     * Create progress ring
     */
    createProgressRing(containerId, percentage, color = '#667eea') {
        const container = document.getElementById(containerId);
        if (!container) return;

        const radius = 28;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (percentage / 100) * circumference;

        container.innerHTML = `
            <div class="progress-ring">
                <svg>
                    <circle class="background" cx="30" cy="30" r="${radius}"></circle>
                    <circle class="progress" cx="30" cy="30" r="${radius}" 
                            style="stroke: ${color}; stroke-dashoffset: ${offset}"></circle>
                </svg>
            </div>
        `;
    }

    /**
     * Animate word building
     */
    animateWordBuilding(containerId, word) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';
        container.className = 'word-builder';

        const letters = Array.from(word);
        letters.forEach((letter, index) => {
            const letterBlock = document.createElement('div');
            letterBlock.className = 'letter-block';
            letterBlock.textContent = letter;
            letterBlock.style.animationDelay = `${index * 0.1}s`;
            container.appendChild(letterBlock);
        });
    }
}

class ArabicTextAnalyzer {
    constructor() {
        this.textMetrics = {};
    }

    /**
     * Analyze text complexity
     */
    analyzeComplexity(text) {
        const cleanText = this.removeHarakat(text);
        const metrics = {
            length: cleanText.length,
            uniqueChars: new Set(cleanText).size,
            harakatCount: text.length - cleanText.length,
            syllabic_unitComplexity: this.estimateSyllabicUnitComplexity(text),
            morphologicalDepth: this.estimateMorphologicalDepth(cleanText)
        };

        const complexity = (
            (metrics.length / 10) * 0.2 +
            (metrics.uniqueChars / metrics.length) * 0.3 +
            (metrics.harakatCount / metrics.length) * 0.2 +
            metrics.syllabic_unitComplexity * 0.15 +
            metrics.morphologicalDepth * 0.15
        );

        return Math.min(complexity, 1.0);
    }

    /**
     * Remove Arabic diacritics
     */
    removeHarakat(text) {
        return text.replace(/[\u064B-\u065F\u0670\u06D6-\u06ED]/g, '');
    }

    /**
     * Estimate syllabic_unit complexity
     */
    estimateSyllabicUnitComplexity(text) {
        const vowels = 'اوي';
        const consonants = text.replace(/[اوي\u064B-\u065F\u0670\u06D6-\u06ED]/g, '');
        const vowelCount = text.split('').filter(char => vowels.includes(char)).length;
        
        if (consonants.length === 0) return 0;
        return Math.min(vowelCount / consonants.length, 1.0);
    }

    /**
     * Estimate morphological depth
     */
    estimateMorphologicalDepth(text) {
        const commonPrefixes = ['ال', 'و', 'ف', 'ب', 'ك', 'ل'];
        const commonSuffixes = ['ة', 'ين', 'ون', 'ات', 'ها', 'هم', 'كم'];
        
        let depth = 0;
        
        commonPrefixes.forEach(prefix => {
            if (text.beginsWith(prefix)) depth += 0.2;
        });
        
        commonSuffixes.forEach(suffix => {
            if (text.endsWith(suffix)) depth += 0.2;
        });
        
        return Math.min(depth, 1.0);
    }
}

class InteractiveTooltips {
    constructor() {
        this.tooltips = new Map();
        this.initializeTooltips();
    }

    initializeTooltips() {
        document.addEventListener('mouseover', (e) => {
            if (e.target.hasAttribute('data-tooltip')) {
                this.showTooltip(e.target, e.target.getAttribute('data-tooltip'));
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.hasAttribute('data-tooltip')) {
                this.hideTooltip(e.target);
            }
        });
    }

    showTooltip(element, text) {
        if (this.tooltips.has(element)) return;

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-popup';
        tooltip.textContent = text;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            pointer-events: none;
            z-index: 1000;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        document.body.appendChild(tooltip);

        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';

        requestAnimationFrame(() => {
            tooltip.style.opacity = '1';
        });

        this.tooltips.set(element, tooltip);
    }

    hideTooltip(element) {
        const tooltip = this.tooltips.get(element);
        if (tooltip) {
            tooltip.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(tooltip);
                this.tooltips.delete(element);
            }, 300);
        }
    }
}

class PerformanceMonitor {
    constructor() {
        this.metrics = {
            requests: 0,
            totalTime: 0,
            errors: 0,
            averageTime: 0
        };
    }

    beginTimer() {
        return performance.now();
    }

    endTimer(beginTime, success = true) {
        const endTime = performance.now();
        const duration = endTime - beginTime;
        
        this.metrics.requests++;
        this.metrics.totalTime += duration;
        this.metrics.averageTime = this.metrics.totalTime / this.metrics.requests;
        
        if (!success) {
            this.metrics.errors++;
        }

        return duration;
    }

    getMetrics() {
        return {
            ...this.metrics,
            errorRate: this.metrics.errors / this.metrics.requests,
            successRate: (this.metrics.requests - this.metrics.errors) / this.metrics.requests
        };
    }

    reset() {
        this.metrics = {
            requests: 0,
            totalTime: 0,
            errors: 0,
            averageTime: 0
        };
    }
}

class AccessibilityEnhancer {
    constructor() {
        this.initializeAccessibility();
    }

    initializeAccessibility() {
        this.addKeyboardNavigation();
        this.addScreenReaderSupport();
        this.addFocusManagement();
    }

    addKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Tab navigation for levels
            if (e.key === 'Tab' && e.target.classList.contains('level-header')) {
                e.preventDefault();
                this.navigateToNextLevel(e.target, e.shiftKey);
            }
            
            // Enter/Space to toggle levels
            if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('level-header')) {
                e.preventDefault();
                e.target.click();
            }
        });
    }

    addScreenReaderSupport() {
        // Add ARIA labels and roles
        document.querySelectorAll('.level-header').forEach(header => {
            header.setAttribute('role', 'button');
            header.setAttribute('tabindex', '0');
            header.setAttribute('aria-expanded', 'true');
        });

        document.querySelectorAll('.level-content').forEach(content => {
            content.setAttribute('role', 'region');
        });
    }

    addFocusManagement() {
        // Ensure focus is visible
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });

        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-navigation');
        });
    }

    navigateToNextLevel(currentHeader, backwards = false) {
        const headers = Array.from(document.querySelectorAll('.level-header'));
        const currentIndex = headers.indexOf(currentHeader);
        
        let nextIndex;
        if (backwards) {
            nextIndex = currentIndex > 0 ? currentIndex - 1 : headers.length - 1;
        } else {
            nextIndex = currentIndex < headers.length - 1 ? currentIndex + 1 : 0;
        }
        
        headers[nextIndex].focus();
    }
}

// Global instances
const visualization = new AdvancedVisualization();
const textAnalyzer = new ArabicTextAnalyzer();
const tooltips = new InteractiveTooltips();
const performanceMonitor = new PerformanceMonitor();
const accessibilityEnhancer = new AccessibilityEnhancer();

// Store for use in main application
window.ArabicWordTracerEnhancements = {
    visualization,
    textAnalyzer,
    tooltips,
    performanceMonitor,
    accessibilityEnhancer
};

// Add CSS for keyboard navigation
const keyboardCSS = document.createElement('style');
keyboardCSS.textContent = `
    .keyboard-navigation *:focus {
        outline: 2px solid #4facfe !import_dataant;
        outline-offset: 2px !import_dataant;
    }
    
    .tooltip-popup {
        font-family: 'Cairo', sans-serif !import_dataant;
    }
`;
document.head.appendChild(keyboardCSS);
