import ast

files = [
    'comprehensive_syntax_batch_fixer.py',
    'emergency_syntax_fixer.py',
    'final_arrow_syntax_fixer.py',
    'simple_arrow_syntax_fixer.py',
    'surgical_syntax_fixer_v2.py',
    'targeted_critical_syntax_fixer.py',
]

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
    ast.parse(file.read())
    print(f'✅ {f}')
    except SyntaxError as e:
    print(f'❌ {f}: Line {e.lineno} - {e.msg}')
        if e.text:
    print(f'   ✂ {e.text.strip()}')
    except Exception as ex:
    print(f'⚠️  {f}: {type(ex).__name__} - {ex}')
