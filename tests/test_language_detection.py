from ingestion.language_detection import detect_language


def test_python_detection():
    assert detect_language("main.py") == "python"


def test_javascript_detection():
    assert detect_language("app.js") == "javascript"


def test_typescript_detection():
    assert detect_language("server.ts") == "typescript"


def test_java_detection():
    assert detect_language("Main.java") == "java"


def test_cpp_detection():
    assert detect_language("main.cpp") == "cpp"


def test_c_detection():
    assert detect_language("main.c") == "c"


def test_go_detection():
    assert detect_language("main.go") == "go"


def test_rust_detection():
    assert detect_language("main.rs") == "rust"


def test_unknown_extension():
    assert detect_language("document.xyz") == "unknown"