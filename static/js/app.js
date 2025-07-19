// Arabic Phonology Engine - Frontend JavaScript

class ArabicPhonologyApp {
    constructor() {
        this.form = document.getElementById('analysisForm');
        this.textInput = document.getElementById('arabicText');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.resultsContainer = document.getElementById('resultsContainer');
        this.welcomeMessage = document.getElementById('welcomeMessage');
        this.errorContainer = document.getElementById('errorContainer');
        this.errorMessage = document.getElementById('errorMessage');
        
        // Add validation and error handling
        this.validateElements();
        this.initializeEventListeners();
    }
    
    /**
     * Validate that all required DOM elements exist
     */
    validateElements() {
        const requiredElements = [
            'form', 'textInput', 'loadingIndicator', 'resultsContainer',
            'welcomeMessage', 'errorContainer', 'errorMessage'
        ];
        
        for (const element of requiredElements) {
            if (!this[element]) {
                throw new Error(`Required element not found: ${element}`);
            }
        }
    }
    
    initializeEventListeners() {
        // Form submission with error handling
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeText();
        });
        
        // Example buttons with improved error handling
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                try {
                    const text = e.target.getAttribute('data-text');
                    if (text) {
                        this.textInput.value = text;
                        this.analyzeText();
                    }
                } catch (error) {
                    console.error('Error handling example button:', error);
                    this.showError('خطأ في تحميل المثال - Error loading example');
                }
            });
        });
        
        // Enhanced keyboard support
        this.textInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.analyzeText();
            }
        });
        
        // Add input validation
        this.textInput.addEventListener('input', (e) => {
            this.validateInput(e.target.value);
        });
    }
    
    /**
     * Validate Arabic text input
     */
    validateInput(text) {
        // Check for Arabic characters
        const arabicRegex = /[\u0600-\u06FF\u0750-\u077F]/;
        const hasArabic = arabicRegex.test(text);
        
        if (text.length > 0 && !hasArabic) {
            this.showError('يرجى إدخال نص عربي - Please enter Arabic text');
            return false;
        }
        
        if (text.length > 1000) {
            this.showError('النص طويل جداً (الحد الأقصى 1000 حرف) - Text too long (max 1000 characters)');
            return false;
        }
        
        this.hideError();
        return true;
    }
    
    async analyzeText() {
        const text = this.textInput.value.trim();
        
        if (!text) {
            this.showError('يرجى إدخال نص للتحليل - Please enter text to analyze');
            return;
        }
        
        if (!this.validateInput(text)) {
            return;
        }
        
        this.showLoading();
        this.hideResults();
        this.hideError();
        
        try {
            const response = await this.makeApiRequest('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.displayResults(data);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`خطأ في التحليل: ${error.message} - Analysis error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    /**
     * Enhanced API request with timeout and retry logic
     */
    async makeApiRequest(url, options, timeout = 10000) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - الطلب تجاوز الوقت المحدد');
            }
            throw error;
        }
    }
    
    displayResults(data) {
        this.hideWelcomeMessage();
        
        try {
            // Display normalized text
            this.displayNormalizedText(data.normalized_text);
            
            // Display phoneme analysis
            this.displayPhonemeAnalysis(data.phoneme_analysis);
            
            // Display syllable encoding
            this.displaySyllableEncoding(data.syllable_encoding);
            
            this.showResults();
        } catch (error) {
            console.error('Error displaying results:', error);
            this.showError('خطأ في عرض النتائج - Error displaying results');
        }
    }
    
    displayNormalizedText(normalizedText) {
        const container = document.getElementById('normalizedText');
        if (!container) {
            console.warn('normalizedText container not found');
            return;
        }
        
        container.innerHTML = `
            <div class="text-center">
                <strong>النص الأصلي:</strong> <span class="original-text">${this.escapeHtml(this.textInput.value)}</span><br>
                <strong>النص المعياري:</strong> <span class="normalized-text">${this.escapeHtml(normalizedText || 'غير متوفر')}</span>
            </div>
        `;
    }
    
    displayPhonemeAnalysis(phonemeAnalysis) {
        const container = document.getElementById('phonemeAnalysis');
        if (!container) {
            console.warn('phonemeAnalysis container not found');
            return;
        }
        
        container.innerHTML = '';
        
        if (!phonemeAnalysis || phonemeAnalysis.length === 0) {
            container.innerHTML = '<p class="text-muted">لا توجد بيانات للتحليل الصوتي - No phoneme analysis data</p>';
            return;
        }
        
        phonemeAnalysis.forEach((item, index) => {
            const [char, data] = item;
            
            // FIXED: Better handling of whitespace and empty characters
            if (!char || (char.trim() === '' && char !== ' ')) {
                return; // Skip truly empty characters but allow spaces
            }
            
            const morphClass = data?.morph_class || 'unknown';
            const cardHtml = this.createPhonemeCard(char, data, morphClass, index);
            container.innerHTML += cardHtml;
        });
    }
    
    /**
     * Create phoneme card HTML with better error handling
     */
    createPhonemeCard(char, data, morphClass, index) {
        const safeData = data || {};
        
        return `
            <div class="col-md-6 col-lg-4 slide-in" style="animation-delay: ${index * 0.1}s">
                <div class="phoneme-card">
                    <div class="phoneme-letter">${this.escapeHtml(char)}</div>
                    <div class="phoneme-property">
                        <span class="property-label">النوع:</span>
                        <span class="property-value">${this.escapeHtml(safeData.type || 'غير محدد')}</span>
                    </div>
                    <div class="phoneme-property">
                        <span class="property-label">المكان:</span>
                        <span class="property-value">${this.escapeHtml(safeData.place || 'غير محدد')}</span>
                    </div>
                    <div class="phoneme-property">
                        <span class="property-label">الطريقة:</span>
                        <span class="property-value">${this.escapeHtml(safeData.manner_primary || 'غير محدد')}</span>
                    </div>
                    <div class="phoneme-property">
                        <span class="property-label">الجهر:</span>
                        <span class="property-value">${this.escapeHtml(safeData.voicing || 'غير محدد')}</span>
                    </div>
                    <div class="phoneme-property">
                        <span class="property-label">التفخيم:</span>
                        <span class="property-value">${safeData.emphatic ? 'مفخم' : 'مرقق'}</span>
                    </div>
                    <div class="phoneme-property">
                        <span class="property-label">التصنيف:</span>
                        <span class="classification-badge classification-${morphClass}">${this.getClassificationName(morphClass)}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    displaySyllableEncoding(syllableEncoding) {
        const container = document.getElementById('syllableEncoding');
        if (!container) {
            console.warn('syllableEncoding container not found');
            return;
        }
        
        if (!syllableEncoding || syllableEncoding.length === 0) {
            container.innerHTML = '<p class="text-muted">لا توجد بيانات لترميز المقاطع - No syllable encoding data</p>';
            return;
        }
        
        let tableHtml = `
            <table class="table syllable-table">
                <thead>
                    <tr>
                        <th>الحرف</th>
                        <th>الحركة</th>
                        <th>رمز المقطع</th>
                        <th>التصنيف</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        syllableEncoding.forEach((unit, index) => {
            const morphClass = unit?.classification || 'unknown';
            tableHtml += `
                <tr class="fade-in" style="animation-delay: ${index * 0.05}s">
                    <td><span class="syllable-letter">${this.escapeHtml(unit?.letter || '')}</span></td>
                    <td>${this.escapeHtml(unit?.vowel || '-')}</td>
                    <td><span class="syllable-code">${this.escapeHtml(unit?.syllable_code || 'غير محدد')}</span></td>
                    <td><span class="classification-badge classification-${morphClass}">${this.getClassificationName(morphClass)}</span></td>
                </tr>
            `;
        });
        
        tableHtml += '</tbody></table>';
        container.innerHTML = tableHtml;
    }
    
    getClassificationName(morphClass) {
        const names = {
            'core': 'أساسي',
            'extra': 'إضافي', 
            'functional': 'وظيفي',
            'weak': 'ضعيف',
            'unknown': 'غير محدد'
        };
        return names[morphClass] || 'غير محدد';
    }
    
    /**
     * HTML escape function for security
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showLoading() {
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = 'block';
            this.loadingIndicator.setAttribute('aria-hidden', 'false');
        }
    }
    
    hideLoading() {
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = 'none';
            this.loadingIndicator.setAttribute('aria-hidden', 'true');
        }
    }
    
    showResults() {
        if (this.resultsContainer) {
            this.resultsContainer.style.display = 'block';
            this.resultsContainer.classList.add('fade-in');
            this.resultsContainer.setAttribute('aria-hidden', 'false');
        }
    }
    
    hideResults() {
        if (this.resultsContainer) {
            this.resultsContainer.style.display = 'none';
            this.resultsContainer.classList.remove('fade-in');
            this.resultsContainer.setAttribute('aria-hidden', 'true');
        }
    }
    
    hideWelcomeMessage() {
        if (this.welcomeMessage) {
            this.welcomeMessage.style.display = 'none';
        }
    }
    
    showError(message) {
        if (this.errorMessage && this.errorContainer) {
            this.errorMessage.textContent = message;
            this.errorContainer.style.display = 'block';
            this.errorContainer.setAttribute('aria-hidden', 'false');
            this.hideWelcomeMessage();
            
            // Auto-hide error after 5 seconds
            setTimeout(() => {
                this.hideError();
            }, 5000);
        }
    }
    
    hideError() {
        if (this.errorContainer) {
            this.errorContainer.style.display = 'none';
            this.errorContainer.setAttribute('aria-hidden', 'true');
        }
    }
}

// Enhanced health check function with retry logic
async function checkServerHealth(retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch('/api/health', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Add timeout
                signal: AbortSignal.timeout(5000)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Server health:', data);
            return data.status === 'healthy';
        } catch (error) {
            console.error(`Health check attempt ${i + 1} failed:`, error);
            if (i === retries - 1) {
                return false;
            }
            // Wait before retry
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    return false;
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Arabic Phonology Engine - Frontend Loading...');
    
    try {
        // Check server health
        const isHealthy = await checkServerHealth();
        if (!isHealthy) {
            console.warn('⚠️ Server health check failed');
            // Show user-friendly message
            const errorContainer = document.getElementById('errorContainer');
            const errorMessage = document.getElementById('errorMessage');
            if (errorContainer && errorMessage) {
                errorMessage.textContent = 'خطأ في الاتصال بالخادم - Server connection error';
                errorContainer.style.display = 'block';
            }
        }
        
        // Initialize the main application
        const app = new ArabicPhonologyApp();
        
        // Make app globally accessible for debugging
        window.arabicPhonologyApp = app;
        
        console.log('✅ Arabic Phonology Engine - Frontend Ready!');
        
        // Add helpful console messages
        console.log('💡 Tips:');
        console.log('  - Use Ctrl+Enter to analyze text quickly');
        console.log('  - Try the example buttons for quick testing');
        console.log('  - Make sure to include diacritics for better analysis');
        console.log('  - Maximum text length: 1000 characters');
        
    } catch (error) {
        console.error('❌ Failed to initialize application:', error);
        
        // Show error to user
        const errorContainer = document.getElementById('errorContainer');
        const errorMessage = document.getElementById('errorMessage');
        if (errorContainer && errorMessage) {
            errorMessage.textContent = 'خطأ في تحميل التطبيق - Application loading error';
            errorContainer.style.display = 'block';
        }
    }
});

// Export for potential external use
window.ArabicPhonologyApp = ArabicPhonologyApp;
