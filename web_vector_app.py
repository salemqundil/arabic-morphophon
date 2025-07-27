#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌐 تطبيق ويب لمولّد المتجه الرقمي للكلمات العربية
==============================================

واجهة ويب تفاعلية لاستخدام نظام توليد المتجهات الرقمية
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from flask import Flask, render_template_string, request, jsonify
import json
from datetime import datetime
from arabic_vector_engine import ArabicDigitalVectorGenerator

app = Flask(__name__)

# إنشاء مولّد المتجه العالمي
generator = ArabicDigitalVectorGenerator()

# قالب HTML للواجهة
HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مولّد المتجه الرقمي للكلمات العربية</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            direction: rtl;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #333;
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .header p {
            color: #666;
            font-size: 1.2em;
            margin: 10px 0;
        }
        
        .input-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border: 2px solid #e9ecef;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        
        input[type="text"], select {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1.1em;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .results {
            display: none;
            margin-top: 30px;
        }
        
        .result-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            border-right: 5px solid #667eea;
        }
        
        .result-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .feature-item {
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 6px;
            border-right: 3px solid #667eea;
        }
        
        .feature-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 5px;
        }
        
        .feature-value {
            color: #333;
            font-size: 1.1em;
        }
        
        .vector-display {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            direction: ltr;
            text-align: left;
        }
        
        .import_dataing {
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-size: 1.2em;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border: 1px solid #f5c6cb;
        }
        
        .features-list {
            background: #e7f3ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .features-list h3 {
            color: #0066cc;
            margin-top: 0;
            font-size: 1.3em;
        }
        
        .features-list ul {
            list-style: none;
            padding: 0;
        }
        
        .features-list li {
            padding: 5px 0;
            padding-right: 20px;
            position: relative;
        }
        
        .features-list li:before {
            content: "✅";
            position: absolute;
            right: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 مولّد المتجه الرقمي للكلمات العربية</h1>
            <p>نظام متقدم لتحليل الكلمات العربية وتوليد المتجهات الرقمية مع الميزات اللغوية الشاملة</p>
        </div>
        
        <div class="features-list">
            <h3>🎯 الميزات المُنفّذة</h3>
            <ul>
                <li>التعيين المعرفي والنكرة والعلم</li>
                <li>حالة الاسم والإعراب (مرفوع/منصوب/مجرور)</li>
                <li>قواعد إدغام اللام مع الحروف الشمسية والقمرية</li>
                <li>حالة الإضافة النحوية والجندر والاتفاق الصرفي</li>
                <li>التصغير وأنماطه المختلفة</li>
                <li>التوزيع الصوتي واللحني والتصريف الشاذ</li>
                <li>العلاقات الدلالية والنمذجة التنبؤية</li>
            </ul>
        </div>
        
        <div class="input-section">
            <form id="analysisForm">
                <div class="form-group">
                    <label for="word">🔤 الكلمة العربية:</label>
                    <input type="text" id="word" name="word" placeholder="مثال: الكتاب، مدرسة، كُتَيْب..." required>
                </div>
                
                <div class="form-group">
                    <label for="context">🎯 السياق النحوي (اختياري):</label>
                    <select id="context" name="context">
                        <option value="">-- اختر السياق --</option>
                        <option value="agent">فاعل (agent)</option>
                        <option value="patient">مفعول (patient)</option>
                        <option value="instrument">أداة (instrument)</option>
                        <option value="location">مكان (location)</option>
                        <option value="time">زمان (time)</option>
                        <option value="manner">طريقة (manner)</option>
                    </select>
                </div>
                
                <button type="submit" class="btn" id="analyzeBtn">⚡ تحليل الكلمة</button>
            </form>
        </div>
        
        <div id="results" class="results">
            <div class="result-card">
                <div class="result-title">📋 الملخص اللغوي</div>
                <div id="linguisticSummary" class="feature-grid"></div>
            </div>
            
            <div class="result-card">
                <div class="result-title">🔢 إحصائيات المتجه</div>
                <div id="vectorStats" class="feature-grid"></div>
            </div>
            
            <div class="result-card">
                <div class="result-title">🔬 الميزات المتقدمة</div>
                <div id="advancedFeatures" class="feature-grid"></div>
            </div>
            
            <div class="result-card">
                <div class="result-title">🎲 عينة من المتجه الرقمي</div>
                <div id="vectorSample" class="vector-display"></div>
            </div>
        </div>
        
        <div id="import_dataing" class="import_dataing" style="display:none;">
            ⏳ جاري تحليل الكلمة...
        </div>
        
        <div id="error" class="error" style="display:none;"></div>
    </div>

    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const word = document.getElementById('word').value.trim();
            const context = document.getElementById('context').value;
            
            if (!word) {
                showError('يرجى إدخال كلمة صالحة');
                return;
            }
            
            // إظهار مؤشر التحميل
            document.getElementById('import_dataing').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        word: word,
                        context: context || null
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    displayResults(result.data);
                } else {
                    showError(result.error || 'حدث خطأ في التحليل');
                }
                
            } catch (error) {
                showError('حدث خطأ في الاتصال: ' + error.message);
            } finally {
                document.getElementById('import_dataing').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        });
        
        function displayResults(data) {
            // الملخص اللغوي
            const summary = data.linguistic_analysis;
            const summaryHTML = Object.entries(summary).map(([key, value]) => 
                `<div class="feature-item">
                    <div class="feature-label">${key}</div>
                    <div class="feature-value">${value}</div>
                </div>`
            ).join('');
            document.getElementById('linguisticSummary').innerHTML = summaryHTML;
            
            // إحصائيات المتجه
            const vector = data.numerical_vector;
            const statsHTML = [
                { label: 'إجمالي الأبعاد', value: vector.length },
                { label: 'أصغر قيمة', value: Math.min(...vector).toFixed(3) },
                { label: 'أكبر قيمة', value: Math.max(...vector).toFixed(3) },
                { label: 'المتوسط', value: (vector.reduce((a,b) => a+b, 0) / vector.length).toFixed(3) }
            ].map(stat => 
                `<div class="feature-item">
                    <div class="feature-label">${stat.label}</div>
                    <div class="feature-value">${stat.value}</div>
                </div>`
            ).join('');
            document.getElementById('vectorStats').innerHTML = statsHTML;
            
            // الميزات المتقدمة
            const components = data.vector_components;
            const advancedHTML = [
                { label: 'نسبة الصوامت', value: components.consonant_ratio.toFixed(3) },
                { label: 'نسبة التفخيم', value: components.emphatic_ratio.toFixed(3) },
                { label: 'التعقد الصرفي', value: components.morphological_complexity.toFixed(3) },
                { label: 'الملموسية', value: components.concreteness.toFixed(3) },
                { label: 'عدد المقاطع', value: components.syllable_count },
                { label: 'طول الجذر', value: components.root_length }
            ].map(feature => 
                `<div class="feature-item">
                    <div class="feature-label">${feature.label}</div>
                    <div class="feature-value">${feature.value}</div>
                </div>`
            ).join('');
            document.getElementById('advancedFeatures').innerHTML = advancedHTML;
            
            // عينة من المتجه
            const sample = vector.slice(0, 20).map(x => x.toFixed(3)).join(', ');
            document.getElementById('vectorSample').innerHTML = 
                `العناصر الأولى (20 عنصر): [${sample}...]`;
            
            // إظهار النتائج
            document.getElementById('results').style.display = 'block';
        }
        
        function showError(message) {
            document.getElementById('error').innerHTML = '❌ ' + message;
            document.getElementById('error').style.display = 'block';
        }
        
        // أمثلة سريعة
        const examples = ['الكتاب', 'مدرسة', 'كُتَيْب', 'مُدرِّس', 'الشمس'];
        let exampleIndex = 0;
        
        setInterval(() => {
            const wordInput = document.getElementById('word');
            if (!wordInput.value) {
                wordInput.placeholder = `مثال: ${examples[exampleIndex]}`;
                exampleIndex = (exampleIndex + 1) % examples.length;
            }
        }, 3000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """الصفحة الرئيسية"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/analyze", methods=["POST"])
def analyze_word():
    """تحليل الكلمة وإرجاع النتائج"""
    try:
        data = request.get_json()
        word = data.get("word", "").strip()
        context_role = data.get("context")

        if not word:
            return jsonify({"status": "error", "error": "يرجى إدخال كلمة صالحة"})

        # تحضير السياق
        context = None
        if context_role:
            context = {"semantic_role": context_role}

        # تحليل الكلمة
        result = generator.generate_vector(word, context)

        if result["processing_status"] == "success":
            return jsonify({"status": "success", "data": result})
        else:
            return jsonify(
                {"status": "error", "error": result.get("error", "فشل في التحليل")}
            )

    except Exception as e:
        return jsonify({"status": "error", "error": f"خطأ في الخادم: {str(e)}"})


@app.route("/health")
def health_check():
    """فحص حالة الخادم"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "generator_ready": True,
        }
    )


if __name__ == "__main__":
    print("🚀 بدء خادم مولّد المتجه الرقمي للكلمات العربية")
    print("🌐 الرابط: http://localhost:5000")
    print("💡 للإيقاف: اضغط Ctrl+C")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reimport_dataer=False,  # لتجنب إعادة تحميل المولّد
    )
