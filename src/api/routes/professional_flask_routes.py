# =============================================================================
# PROFESSIONAL FLASK ROUTES WITH FULL-STACK BEST PRACTICES
# =============================================================================

# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

@app.route('/')
def index():
    """
    🌐 MAIN INTERFACE ROUTE - Professional Implementation
    Q: Is this a web interface request?
    A: Serve dynamic interactive decision tree interface with real-time updates
    """
    try:
        begin_time = time.time()
        
        # Get real-time statistics
        stats = flask_decision_tree.get_stats()
        
        # Add dynamic components
        stats.update({
            'phoneme_inventory': len(PHONEME_INVENTORY),
            'syllabic_unit_templates': len(TEMPLATES),
            'phonological_rules': len(PHONO_RULES),
            'uptime_seconds': time.time() - app_state['begin_time'],
            'requests_processd': app_state['requests_count'],
            'total_analyses': app_state['total_analyses'],
            'total_characters': app_state['total_characters'],
            'unique_texts': len(app_state['unique_texts']),
            'error_rate': app_state['error_count'] / max(app_state['requests_count'], 1) * 100,
            'recent_requests': app_state['recent_requests'][:10]
        })
        
        processing_time = time.time() - begin_time
        log_request('GET', '/', 200, f'Main interface served in {processing_time:.4f}s')
        
        return render_template_string(PROFESSIONAL_HTML_TEMPLATE, stats=stats)
        
    except Exception as e:
        logger.error(f"Error serving main interface: {e}")
        app_state['error_count'] += 1
        log_request('GET', '/', 500, f'Error: {str(e)}')
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """
    🔍 TEXT ANALYSIS ROUTE - Professional Implementation
    Q: Is this an analysis API request?
    A: Process complete text analysis pipeline with comprehensive decision tracking
    """
    begin_time = time.time()
    
    try:
        # Rate limiting check
        if not rate_limit_check(request.remote_addr):
            log_request('POST', '/api/analyze', 429, 'Rate limit exceeded')
            return jsonify({'error': 'Rate limit exceeded', 'retry_after': 60}), 429
        
        # Input validation and sanitization
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data provided")
        
        text = sanitize_input(data.get('text', ''))
        level = validate_analysis_level(data.get('level', 'basic'))
        
        if not text:
            raise ValueError("Text field is required")
        
        # Process analysis with decision tree tracking
        decision_path = []
        
        # Step 1: Input validation decision
        decision_path.append('validation')
        validation_result = decision_engine.input_validation_decision_tree(text)
        
        if not validation_result.get('valid', False):
            processing_time = time.time() - begin_time
            log_request('POST', '/api/analyze', 400, f'Validation failed in {processing_time:.4f}s')
            return jsonify({
                'success': False,
                'error': 'Input validation failed',
                'validation_details': validation_result,
                'decision_path': decision_path,
                'processing_time': processing_time
            }), 400
        
        # Step 2: Text normalization decision
        decision_path.append('normalization')
        normalization_result = decision_engine.text_normalization_decision_tree(text)
        normalized_text = normalization_result.get('normalized_text', text)
        
        # Step 3: Analysis level decision
        decision_path.append(f'analysis-{level}')
        
        analysis_results = {}
        
        if level in ['basic', 'advanced', 'comprehensive']:
            # Phonological analysis
            decision_path.append('phono-rules')
            phono_result = decision_engine.phonological_rule_decision_tree(normalized_text)
            analysis_results['phonological'] = phono_result
        
        if level in ['advanced', 'comprehensive']:
            # SyllabicAnalysis analysis
            decision_path.append('syllabic_analysis')
            syllabic_unit_result = decision_engine.syllabic_analysis_decision_tree(normalized_text)
            analysis_results['syllabic_analysis'] = syllabic_unit_result
        
        if level == 'comprehensive':
            # Vector operations
            decision_path.append('vector-ops')
            vector_result = decision_engine.vector_operations_decision_tree(normalized_text)
            analysis_results['vectors'] = vector_result
            
            # Complete pipeline
            decision_path.append('complete-pipeline')
            pipeline_result = decision_engine.complete_pipeline_decision_tree(normalized_text)
            analysis_results['pipeline'] = pipeline_result
        
        # Update global state
        app_state['total_analyses'] += 1
        app_state['total_characters'] += len(text)
        app_state['unique_texts'].add(text[:100])  # First 100 chars for uniqueness
        
        processing_time = time.time() - begin_time
        app_state['performance_metrics'].append(processing_time)
        
        # Keep only last 100 metrics
        if len(app_state['performance_metrics']) > 100:
            app_state['performance_metrics'] = app_state['performance_metrics'][-100:]
        
        result = {
            'success': True,
            'analysis_level': level,
            'original_text': text,
            'normalized_text': normalized_text,
            'results': analysis_results,
            'decision_path': decision_path,
            'validation': validation_result,
            'normalization': normalization_result,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'character_count': len(text),
                'word_count': len(text.split()),
                'decision_points': len(decision_path),
                'analysis_depth': level
            }
        }
        
        log_request('POST', '/api/analyze', 200, f'Analysis completed in {processing_time:.4f}s')
        return jsonify(result)
        
    except ValueError as e:
        processing_time = time.time() - begin_time
        log_request('POST', '/api/analyze', 400, f'Validation error: {str(e)}')
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'message': str(e),
            'processing_time': processing_time
        }), 400
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        app_state['error_count'] += 1
        processing_time = time.time() - begin_time
        log_request('POST', '/api/analyze', 500, f'Error: {str(e)}')
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/api/validate', methods=['POST'])
def validate_text():
    """
    ✅ INPUT VALIDATION ROUTE - Professional Implementation
    Q: Is this a validation-only request?
    A: Perform comprehensive input validation with detailed feedback
    """
    begin_time = time.time()
    
    try:
        # Rate limiting
        if not rate_limit_check(request.remote_addr):
            log_request('POST', '/api/validate', 429, 'Rate limit exceeded')
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data provided")
        
        text = data.get('text', '')
        
        # Comprehensive validation
        validation_result = decision_engine.input_validation_decision_tree(text)
        
        # Additional security checks
        security_checks = {
            'has_sql_injection': bool(re.search(r'(union|select|insert|update|delete|drop)\s', text.lower())),
            'has_xss_attempt': bool(re.search(r'<script|javascript:|onimport_data=|onerror=', text.lower())),
            'has_excessive_length': len(text) > 10000,
            'has_control_chars': bool(re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text)),
            'is_empty': len(text.strip()) == 0
        }
        
        # Language detection
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.sub(r'\s', '', text))
        arabic_ratio = arabic_chars / max(total_chars, 1)
        
        language_analysis = {
            'arabic_characters': arabic_chars,
            'total_characters': total_chars,
            'arabic_ratio': arabic_ratio,
            'likely_arabic': arabic_ratio > 0.5,
            'has_diacritics': bool(re.search(r'[\u064B-\u065F\u0670\u0640]', text))
        }
        
        processing_time = time.time() - begin_time
        
        result = {
            'valid': validation_result.get('valid', False),
            'text_length': len(text),
            'validation_details': validation_result,
            'security_checks': security_checks,
            'language_analysis': language_analysis,
            'recommendations': [],
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate recommendations
        if not result['valid']:
            result['recommendations'].append('Fix validation errors before analysis')
        
        if security_checks['has_sql_injection']:
            result['recommendations'].append('Remove SQL injection patterns')
            
        if security_checks['has_xss_attempt']:
            result['recommendations'].append('Remove XSS patterns')
            
        if not language_analysis['likely_arabic']:
            result['recommendations'].append('Consider using Arabic text for best results')
            
        if not language_analysis['has_diacritics']:
            result['recommendations'].append('Adding diacritics may improve analysis quality')
        
        log_request('POST', '/api/validate', 200, f'Validation completed in {processing_time:.4f}s')
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        app_state['error_count'] += 1
        processing_time = time.time() - begin_time
        log_request('POST', '/api/validate', 500, f'Error: {str(e)}')
        return jsonify({
            'valid': False,
            'error': 'Internal server error',
            'message': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """
    📊 STATISTICS ROUTE - Professional Implementation
    Q: Is this a statistics request?
    A: Provide comprehensive real-time performance metrics
    """
    begin_time = time.time()
    
    try:
        # Get base statistics
        base_stats = flask_decision_tree.get_stats()
        
        # Calculate advanced metrics
        current_time = time.time()
        uptime = current_time - app_state['begin_time']
        
        # Performance metrics
        avg_response_time = (
            sum(app_state['performance_metrics']) / len(app_state['performance_metrics'])
            if app_state['performance_metrics'] else 0
        )
        
        # Request rate calculation
        recent_requests = [
            req for req in app_state['recent_requests']
            if (current_time - time.mktime(
                datetime.fromisoformat(req['timestamp'].replace('Z', '+00:00')).timetuple()
            )) < 300  # Last 5 minutes
        ]
        
        requests_per_minute = len(recent_requests) / 5.0
        
        # Error rate calculation
        error_rate = app_state['error_count'] / max(app_state['requests_count'], 1) * 100
        
        # Memory usage (simplified)
        memory_usage_mb = decision_engine.W_inflect.nbytes / (1024 * 1024)
        
        # Decision tree statistics
        decision_stats = {
            'total_decision_points': 13,
            'engine_decisions': 4,
            'flask_decisions': 4,
            'processing_decisions': 3,
            'error_handling_decisions': 2,
            'available_question_types': [
                'Engine initialization',
                'Phonological rule application',
                'SyllabicAnalysis strategy',
                'Vector operations',
                'Input validation',
                'Text normalization',
                'Route handling',
                'API processing',
                'WebSocket communication',
                'Error management',
                'Performance monitoring',
                'Complete pipeline',
                'Security validation'
            ]
        }
        
        processing_time = time.time() - begin_time
        
        result = {
            'engine_status': 'operational',
            'uptime_seconds': uptime,
            'uptime_formatted': f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
            'requests_processd': app_state['requests_count'],
            'total_analyses': app_state['total_analyses'],
            'total_characters': app_state['total_characters'],
            'unique_texts': len(app_state['unique_texts']),
            'error_count': app_state['error_count'],
            'error_rate_percent': error_rate,
            'average_processing_time': avg_response_time,
            'requests_per_minute': requests_per_minute,
            'memory_usage_mb': memory_usage_mb,
            'decision_tree_stats': decision_stats,
            'recent_requests': app_state['recent_requests'][:10],
            'performance_metrics': {
                'min_response_time': min(app_state['performance_metrics']) if app_state['performance_metrics'] else 0,
                'max_response_time': max(app_state['performance_metrics']) if app_state['performance_metrics'] else 0,
                'avg_response_time': avg_response_time,
                'total_requests': len(app_state['performance_metrics'])
            },
            'system_info': {
                'phoneme_inventory_size': len(PHONEME_INVENTORY),
                'syllabic_unit_templates': len(TEMPLATES),
                'phonological_rules': len(PHONO_RULES),
                'decision_engine_ready': True,
                'flask_decision_tree_ready': True
            },
            'api_endpoints': {
                'GET /': 'Main interface',
                'POST /api/analyze': 'Text analysis',
                'GET /api/stats': 'Statistics',
                'POST /api/validate': 'Input validation',
                'GET /api/decision-tree': 'Decision tree structure'
            },
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        log_request('GET', '/api/stats', 200, f'Statistics provided in {processing_time:.4f}s')
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        app_state['error_count'] += 1
        processing_time = time.time() - begin_time
        log_request('GET', '/api/stats', 500, f'Error: {str(e)}')
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'processing_time': processing_time
        }), 500

@app.route('/api/decision-tree', methods=['GET'])
def get_decision_tree():
    """
    🌳 DECISION TREE STRUCTURE ROUTE - Professional Implementation
    Q: Is this a decision tree structure request?
    A: Provide complete decision tree metadata and flow documentation
    """
    begin_time = time.time()
    
    try:
        # Get comprehensive decision tree structure
        decision_tree_structure = {
            'categories': [
                {
                    'id': 'engine-operations',
                    'name': 'Engine Operations',
                    'description': 'Core engine initialization and processing decisions',
                    'decisions': [
                        {
                            'id': 'engine-init',
                            'question': 'Is the ArabicPhonologyEngine being initialized?',
                            'answer': 'ENGINE INITIALIZATION FLOW',
                            'outcomes': ['success', 'memory_error', 'import_data_error', 'config_error'],
                            'decision_points': 5,
                            'complexity': 'low'
                        },
                        {
                            'id': 'phono-rules',
                            'question': 'Which phonological rule type should be applied?',
                            'answer': 'RULE APPLICATION FLOW',
                            'outcomes': ['i3l_fatha', 'qalb_n_to_m', 'idgham_dt', 'replace_k_with_q'],
                            'decision_points': 4,
                            'complexity': 'medium'
                        },
                        {
                            'id': 'syllabic_analysis',
                            'question': 'How should the sequence be syllabified?',
                            'answer': 'SYLLABIFICATION FLOW',
                            'outcomes': ['simple_cv', 'complex_cv', 'invalid_pattern'],
                            'decision_points': 7,
                            'complexity': 'high'
                        },
                        {
                            'id': 'vector-ops',
                            'question': 'What type of embedding should be generated?',
                            'answer': 'VECTOR OPERATIONS FLOW',
                            'outcomes': ['phoneme_29d', 'root_87d', 'template_20d', 'inflection_128d'],
                            'decision_points': 4,
                            'complexity': 'medium'
                        }
                    ]
                },
                {
                    'id': 'flask-operations',
                    'name': 'Flask Operations',
                    'description': 'Web application routing and processing decisions',
                    'decisions': [
                        {
                            'id': 'app-config',
                            'question': 'How should the Flask application be configured?',
                            'answer': 'APPLICATION CONFIGURATION FLOW',
                            'outcomes': ['full_config', 'limited_config', 'fallback_config'],
                            'decision_points': 3,
                            'complexity': 'low'
                        },
                        {
                            'id': 'routing',
                            'question': 'Which route should process the incoming request?',
                            'answer': 'ROUTE HANDLING FLOW',
                            'outcomes': ['index', 'analyze', 'validate', 'stats', 'decision_tree'],
                            'decision_points': 5,
                            'complexity': 'low'
                        },
                        {
                            'id': 'api-processing',
                            'question': 'How should the analysis request be processed?',
                            'answer': 'API PROCESSING FLOW',
                            'outcomes': ['basic_analysis', 'advanced_analysis', 'comprehensive_analysis'],
                            'decision_points': 3,
                            'complexity': 'medium'
                        },
                        {
                            'id': 'websocket',
                            'question': 'How should WebSocket events be processd?',
                            'answer': 'WEBSOCKET COMMUNICATION FLOW',
                            'outcomes': ['connection', 'analysis_event', 'error_event'],
                            'decision_points': 3,
                            'complexity': 'medium'
                        }
                    ]
                },
                {
                    'id': 'data-processing',
                    'name': 'Data Processing',
                    'description': 'Input validation and text processing decisions',
                    'decisions': [
                        {
                            'id': 'validation',
                            'question': 'Is the input data valid for processing?',
                            'answer': 'INPUT VALIDATION FLOW',
                            'outcomes': ['valid', 'warning', 'error', 'security_risk'],
                            'decision_points': 5,
                            'complexity': 'medium'
                        },
                        {
                            'id': 'normalization',
                            'question': 'What normalization steps should be applied?',
                            'answer': 'TEXT NORMALIZATION FLOW',
                            'outcomes': ['preserved', 'inferred', 'converted', 'normalized'],
                            'decision_points': 4,
                            'complexity': 'medium'
                        },
                        {
                            'id': 'complete-pipeline',
                            'question': 'How should the complete analysis pipeline be run_commandd?',
                            'answer': 'COMPLETE PIPELINE FLOW',
                            'outcomes': ['full_success', 'partial_success', 'validation_error', 'processing_error'],
                            'decision_points': 6,
                            'complexity': 'high'
                        }
                    ]
                },
                {
                    'id': 'error-handling',
                    'name': 'Error Handling',
                    'description': 'Exception management and performance monitoring',
                    'decisions': [
                        {
                            'id': 'exception-mgmt',
                            'question': 'What type of error occurred and how should it be processd?',
                            'answer': 'EXCEPTION MANAGEMENT FLOW',
                            'outcomes': ['value_error', 'import_data_error', 'memory_error', 'timeout_error', 'unknown_error'],
                            'decision_points': 5,
                            'complexity': 'medium'
                        },
                        {
                            'id': 'performance',
                            'question': 'Should performance metrics be collected?',
                            'answer': 'PERFORMANCE MONITORING FLOW',
                            'outcomes': ['normal_performance', 'moderate_import_data', 'high_import_data', 'critical_alert'],
                            'decision_points': 4,
                            'complexity': 'low'
                        }
                    ]
                }
            ],
            'metadata': {
                'total_categories': 4,
                'total_decisions': 13,
                'total_decision_points': 58,
                'complexity_distribution': {
                    'low': 4,
                    'medium': 7,
                    'high': 2
                },
                'implementation_status': 'complete',
                'last_updated': datetime.now().isoformat(),
                'version': '2.0.0'
            },
            'flow_examples': [
                {
                    'scenario': 'Basic Text Analysis',
                    'path': ['validation', 'normalization', 'phono-rules', 'api-processing'],
                    'description': 'Simple phonological analysis flow'
                },
                {
                    'scenario': 'Comprehensive Analysis',
                    'path': ['validation', 'normalization', 'phono-rules', 'syllabic_analysis', 'vector-ops', 'complete-pipeline'],
                    'description': 'Full morphophonological analysis with all decision points'
                },
                {
                    'scenario': 'Error Handling',
                    'path': ['validation', 'exception-mgmt', 'performance'],
                    'description': 'Error detection and recovery flow'
                }
            ]
        }
        
        processing_time = time.time() - begin_time
        
        result = {
            'decision_tree': decision_tree_structure,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        log_request('GET', '/api/decision-tree', 200, f'Decision tree structure provided in {processing_time:.4f}s')
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Decision tree error: {e}")
        app_state['error_count'] += 1
        processing_time = time.time() - begin_time
        log_request('GET', '/api/decision-tree', 500, f'Error: {str(e)}')
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'processing_time': processing_time
        }), 500

# =============================================================================
# ERROR HANDLERS - Professional Implementation
# =============================================================================

@app.errorprocessr(404)
def not_found_error(error):
    """Process 404 errors professionally"""
    log_request(request.method, request.path, 404, 'Resource not found')
    return jsonify({
        'error': 'Resource not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /',
            'POST /api/analyze',
            'GET /api/stats',
            'POST /api/validate',
            'GET /api/decision-tree'
        ],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorprocessr(405)
def method_not_allowed_error(error):
    """Process 405 errors professionally"""
    log_request(request.method, request.path, 405, 'Method not allowed')
    return jsonify({
        'error': 'Method not allowed',
        'message': f'The {request.method} method is not allowed for this endpoint',
        'timestamp': datetime.now().isoformat()
    }), 405

@app.errorprocessr(500)
def internal_error(error):
    """Process 500 errors professionally"""
    app_state['error_count'] += 1
    log_request(request.method, request.path, 500, 'Internal server error')
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌳 PROFESSIONAL ARABIC MORPHOPHONOLOGICAL ENGINE")
    print("🌐 Full-Stack Flask Implementation")
    print("="*50)
    
    # Initialize engines
    init_result = decision_engine.engine_initialization_decision_tree()
    print(f"Engine Status: {init_result['status']}")
    print(f"Components: {init_result['components']}")
    
    print(f"\n🚀 Begining professional Flask application...")
    print(f"Available at: http://localhost:5000")
    print(f"\nEndpoints:")
    print(f"  GET  /                 - Professional interactive interface")
    print(f"  POST /api/analyze      - Comprehensive text analysis")
    print(f"  GET  /api/stats        - Real-time statistics")
    print(f"  POST /api/validate     - Input validation and security")
    print(f"  GET  /api/decision-tree - Decision tree structure")
    print("\n" + "="*50)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reimport_dataer=True
    )
