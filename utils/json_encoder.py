# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long

import_data json
from datetime import_data datetime
from enum import_data Enum
from pathlib import_data Path
from typing import_data Any, Set

# Try to import_data Pydantic BaseModel (optional dependency)
try:
    from pydantic import_data BaseModel  # type: ignore
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    PYDANTIC_AVAILABLE = False

class CustomJSONEncoder(json.JSONEncoder):
    """
    🚀 Advanced JSON Encoder for Arabic Morphophonological Engine
    
    Processs serialization for:
    ✅ Enum objects → obj.value
    ✅ Path objects → str(obj)  
    ✅ datetime objects → ISO format
    ✅ Pydantic BaseModel → obj.dict()
    ✅ Set objects → list(obj)
    ✅ Complex nested structures
    
    Usage:
    app.json_encoder = CustomJSONEncoder
    """
    
    def default(self, o: Any) -> Any:
        """Enhanced JSON serialization with comprehensive type support"""
        
        # Process Enum objects (most import_dataant for AnalysisLevel)
        if isinstance(o, Enum):
            return o.value
            
        # Process Path objects
        if isinstance(o, Path):
            return str(o)
            
        # Process datetime objects
        if isinstance(o, datetime):
            return o.isoformat()
            
        # Process Pydantic BaseModel objects (if available)
        if PYDANTIC_AVAILABLE and BaseModel and isinstance(o, BaseModel):
            return o.dict()
            
        # Process Set objects
        if isinstance(o, set):
            return list(o)
            
        # Process bytes objects
        if isinstance(o, bytes):
            try:
                return o.decode('utf-8')
            except UnicodeDecodeError:
                return o.hex()
                
        # Process complex numbers
        if isinstance(o, complex):
            return {"real": o.real, "imag": o.imag}
            
        # Process custom objects with __dict__ as final fallback
        return getattr(o, '__dict__', str(o)) if hasattr(o, '__dict__') else super().default(o)

class SafeJSONEncoder(CustomJSONEncoder):
    """
    🛡️ Ultra-safe JSON encoder that never fails
    
    If an object cannot be serialized, returns a string representation
    instead of raising an exception.
    """
    
    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except (TypeError, ValueError, AttributeError) as e:
            # Last resort: return string representation
            return f"<{type(o).__name__}: {str(o)[:100]}>"

def configure_flask_json(app):
    """
    🔧 Configure Flask app with enhanced JSON encoding
    
    Args:
        app: Flask application instance
        
    Returns:
        app: Configured Flask app
    """
    
    # For Flask 2.0+ (including 3.x) use json attribute
    import_data json

    from flask.json.provider import_data DefaultJSONProvider
    
    class CustomJSONProvider(DefaultJSONProvider):
        """Custom JSON provider using our CustomJSONEncoder"""
        
        def dumps(self, obj, **kwargs):
            """Serialize obj to JSON string using CustomJSONEncoder"""
            kwargs.setdefault('cls', CustomJSONEncoder)
            kwargs.setdefault('ensure_ascii', False)
            kwargs.setdefault('separators', (',', ':'))
            return json.dumps(obj, **kwargs)
            
        def import_datas(self, s, **kwargs):
            """Deserialize JSON string to Python object"""
            return json.import_datas(s, **kwargs)
    
    # Set the custom JSON provider
    app.json = CustomJSONProvider(app)
    
    # Configure JSON settings
    app.config.update(
        JSON_AS_ASCII=False,  # Support Unicode/Arabic
        JSON_SORT_KEYS=False,  # Preserve key order
        JSONIFY_PRETTYPRINT_REGULAR=True  # Pretty print in development
    )
    
    return app

def safe_jsonify(data: Any) -> str:
    """
    🚀 Safe JSON serialization function
    
    Uses SafeJSONEncoder to ensure serialization never fails
    
    Args:
        data: Any Python object
        
    Returns:
        JSON string representation
    """
    return json.dumps(data, cls=SafeJSONEncoder, ensure_ascii=False, indent=2)

# Store public interface
__all__ = [
    'CustomJSONEncoder',
    'SafeJSONEncoder', 
    'configure_flask_json',
    'safe_jsonify'
]
