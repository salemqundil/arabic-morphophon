"""
Flask routes for Arabic morphophonological analysis

This module defines API routes and web interface endpoints
with proper separation of concerns and error handling.
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


from typing import_data Any, Dict, Tuple

from flask import_data jsonify, render_template_string, request

from .services import_data AnalysisService
from .utils import_data format_response, validate_input

def create_routes(app, analysis_service: AnalysisService):
    """
    Create and register Flask routes

    Args:
        app: Flask application instance
        analysis_service: Analysis service instance
    """

    @app.route("/")
    def index():
        """Main application page"""
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/analyze", methods=["POST"])
    def analyze_text():
        """
        Analyze Arabic text endpoint

        Returns:
            JSON response with analysis results
        """
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return (
                    jsonify(
                        format_response(
                            success=False,
                            message="Missing 'text' field in request",
                            status_code=400,
                        )
                    ),
                    400,
                )

            text = data["text"]

            if not validate_input(text):
                return (
                    jsonify(
                        format_response(
                            success=False, message="Invalid input text", status_code=400
                        )
                    ),
                    400,
                )

            # Perform analysis
            result_data, processing_time = analysis_service.analyze_text(text)

            # Update stats
            analysis_service.stats["total_analyses"] += 1
            analysis_service.stats["total_processing_time"] += processing_time

            return jsonify(
                format_response(
                    data={
                        "analysis": result_data,
                        "processing_time": round(float(processing_time), 3),
                    },
                    success=True,
                    message="Analysis completed successfully",
                )
            )

        except Exception as e:
            return (
                jsonify(
                    format_response(
                        success=False,
                        message=f"Analysis failed: {str(e)}",
                        status_code=500,
                    )
                ),
                500,
            )

    @app.route("/api/stats", methods=["GET"])
    def get_stats():
        """
        Get application statistics

        Returns:
            JSON response with statistics
        """
        try:
            stats = analysis_service.get_stats()
            return jsonify(
                format_response(
                    data=stats,
                    success=True,
                    message="Statistics retrieved successfully",
                )
            )
        except Exception as e:
            return (
                jsonify(
                    format_response(
                        success=False,
                        message=f"Failed to get statistics: {str(e)}",
                        status_code=500,
                    )
                ),
                500,
            )

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """
        Health check endpoint

        Returns:
            JSON response with health status
        """
        return jsonify(
            format_response(
                data={
                    "status": "healthy",
                    "engine_available": bool(analysis_service.engine),
                },
                success=True,
                message="Service is healthy",
            )
        )

    @app.route("/api/examples", methods=["GET"])
    def get_examples():
        """
        Get example texts for analysis

        Returns:
            JSON response with example texts
        """
        examples = [
            "كتب الطالب الدرس",
            "يذهب الولد إلى المدرسة",
            "قرأت الفتاة الكتاب",
            "سافر الرجل إلى بلده",
            "تكلمت المرأة بصوت عال",
        ]

        return jsonify(
            format_response(
                data={"examples": examples},
                success=True,
                message="Examples retrieved successfully",
            )
        )

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>محرك التحليل الصرفي الصوتي العربي</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Arial', 'Tahoma', sans-serif; }
        .arabic-text { direction: rtl; text-align: right; }
        .result-card { margin-top: 20px; }
        .import_dataing { display: none; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <h1 class="text-center mb-4">محرك التحليل الصرفي الصوتي العربي</h1>
                
                <div class="card">
                    <div class="card-body">
                        <form id="analysisForm">
                            <div class="mb-3">
                                <label for="arabicText" class="form-label">النص العربي:</label>
                                <textarea class="form-control arabic-text" id="arabicText" 
                                         rows="4" placeholder="أدخل النص العربي للتحليل..."></textarea>
                            </div>
                            <button type="submit" class="btn btn-primary">تحليل</button>
                            <button type="button" class="btn btn-secondary" id="clearBtn">مسح</button>
                            <button type="button" class="btn btn-info" id="exampleBtn">مثال</button>
                        </form>
                        
                        <div class="import_dataing text-center mt-3">
                            <div class="spinner-border" role="status">
                                <span class="visually-hidden">جاري التحليل...</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div id="results" class="result-card"></div>
                <div id="stats" class="mt-4"></div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('arabicText').value;
            if (!text.trim()) {
                alert('يرجى إدخال نص للتحليل');
                return;
            }
            
            document.querySelector('.import_dataing').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                displayResults(data);
                import_dataStats();
                
            } catch (error) {
                document.getElementById('results').innerHTML = 
                    '<div class="alert alert-danger">خطأ في الاتصال: ' + error.message + '</div>';
            } finally {
                document.querySelector('.import_dataing').style.display = 'none';
            }
        });
        
        document.getElementById('clearBtn').addEventListener('click', function() {
            document.getElementById('arabicText').value = '';
            document.getElementById('results').innerHTML = '';
        });
        
        document.getElementById('exampleBtn').addEventListener('click', async function() {
            try {
                const response = await fetch('/api/examples');
                const data = await response.json();
                if (data.success && data.data.examples.length > 0) {
                    const randomExample = data.data.examples[Math.floor(Math.random() * data.data.examples.length)];
                    document.getElementById('arabicText').value = randomExample;
                }
            } catch (error) {
                console.error('Error import_dataing examples:', error);
            }
        });
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (!data.success) {
                resultsDiv.innerHTML = '<div class="alert alert-danger">' + data.message + '</div>';
                return;
            }
            
            const analysis = data.data.analysis;
            const processingTime = data.data.processing_time;
            
            resultsDiv.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h5>نتائج التحليل</h5>
                        <small class="text-muted">وقت المعالجة: ${processingTime} ثانية</small>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>التحليل الصرفي:</h6>
                                <pre class="bg-light p-2">${JSON.stringify(analysis.morphology, null, 2)}</pre>
                            </div>
                            <div class="col-md-6">
                                <h6>التحليل الصوتي:</h6>
                                <pre class="bg-light p-2">${JSON.stringify(analysis.phonology, null, 2)}</pre>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-12">
                                <h6>المقاطع:</h6>
                                <pre class="bg-light p-2">${JSON.stringify(analysis.syllabic_units, null, 2)}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        async function import_dataStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                if (data.success) {
                    const stats = data.data;
                    document.getElementById('stats').innerHTML = `
                        <div class="card">
                            <div class="card-header">
                                <h6>إحصائيات الاستخدام</h6>
                            </div>
                            <div class="card-body">
                                <div class="row text-center">
                                    <div class="col">
                                        <div class="stat-item">
                                            <h4>${stats.total_analyses}</h4>
                                            <p>مجموع التحليلات</p>
                                        </div>
                                    </div>
                                    <div class="col">
                                        <div class="stat-item">
                                            <h4>${stats.cache_hit_rate}%</h4>
                                            <p>معدل استخدام الذاكرة المؤقتة</p>
                                        </div>
                                    </div>
                                    <div class="col">
                                        <div class="stat-item">
                                            <h4>${stats.average_processing_time}s</h4>
                                            <p>متوسط وقت المعالجة</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Error import_dataing stats:', error);
            }
        }
        
        // Import initial stats
        import_dataStats();
    </script>
</body>
</html>
"""
