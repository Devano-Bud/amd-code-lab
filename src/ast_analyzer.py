"""
AST-based code analysis using tree-sitter.

Extracts structural info from Python and JavaScript code:
- Function definitions and signatures
- Class hierarchies
- Import dependencies
- Complexity metrics (nesting depth, function length)
- Variable usage patterns

Usage:
    python ast_analyzer.py <file.py>
    python ast_analyzer.py --dir ./src/
    python ast_analyzer.py --code "def foo(): pass"
"""

import os
import sys
import argparse
import json
from pathlib import Path

try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    from tree_sitter import Language, Parser
except ImportError:
    print("tree-sitter not installed. Run: pip install tree-sitter tree-sitter-python tree-sitter-javascript")
    sys.exit(1)


class CodeAnalyzer:
    """Analyze code structure using tree-sitter ASTs."""

    def __init__(self):
        self.python_lang = Language(tspython.language())
        self.js_lang = Language(tsjavascript.language())
        self.parser = Parser()

    def set_language(self, filename):
        """Set parser language based on file extension."""
        ext = Path(filename).suffix.lower()
        if ext == ".py":
            self.parser.language = self.python_lang
            return "python"
        elif ext in (".js", ".jsx", ".mjs"):
            self.parser.language = self.js_lang
            return "javascript"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def parse(self, code, lang="python"):
        """Parse code string and return AST."""
        if lang == "python":
            self.parser.language = self.python_lang
        else:
            self.parser.language = self.js_lang

        if isinstance(code, str):
            code = bytes(code, "utf8")
        return self.parser.parse(code)

    def extract_functions(self, tree, lang="python"):
        """Extract all function/method definitions."""
        functions = []
        root = tree.root_node

        if lang == "python":
            query_str = """
            (function_definition
                name: (identifier) @name
                parameters: (parameters) @params
                body: (block) @body
            ) @func
            """
        else:
            query_str = """
            (function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body
            ) @func
            """

        # manual walk since tree_sitter query API varies
        self._walk_for_functions(root, functions, lang)
        return functions

    def _walk_for_functions(self, node, functions, lang):
        """Recursively walk AST looking for function defs."""
        if lang == "python" and node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            body_node = node.child_by_field_name("body")

            func_info = {
                "name": name_node.text.decode() if name_node else "anonymous",
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "params": self._extract_python_params(params_node) if params_node else [],
                "length": (node.end_point[0] - node.start_point[0]) + 1,
                "nesting_depth": self._calc_nesting_depth(body_node) if body_node else 0,
                "is_method": self._is_method(node),
                "decorators": self._extract_decorators(node),
            }
            functions.append(func_info)

        elif lang == "javascript" and node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            params_node = node.child_by_field_name("parameters")
            body_node = node.child_by_field_name("body")

            func_info = {
                "name": name_node.text.decode() if name_node else "anonymous",
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "params": self._extract_js_params(params_node) if params_node else [],
                "length": (node.end_point[0] - node.start_point[0]) + 1,
                "nesting_depth": self._calc_nesting_depth(body_node) if body_node else 0,
                "is_method": False,
                "decorators": [],
            }
            functions.append(func_info)

        for child in node.children:
            self._walk_for_functions(child, functions, lang)

    def extract_classes(self, tree, lang="python"):
        """Extract class definitions with methods."""
        classes = []
        root = tree.root_node

        if lang != "python":
            # JS classes are different, just track for python for now
            return classes

        self._walk_for_classes(root, classes)
        return classes

    def _walk_for_classes(self, node, classes):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            body_node = node.child_by_field_name("body")

            methods = []
            if body_node:
                for child in body_node.children:
                    if child.type == "function_definition":
                        mname = child.child_by_field_name("name")
                        methods.append(mname.text.decode() if mname else "?")

            classes.append({
                "name": name_node.text.decode() if name_node else "anonymous",
                "line": node.start_point[0] + 1,
                "methods": methods,
                "num_methods": len(methods),
            })

        for child in node.children:
            self._walk_for_classes(child, classes)

    def extract_imports(self, tree, lang="python"):
        """Extract import statements."""
        imports = []
        root = tree.root_node
        self._walk_for_imports(root, imports, lang)
        return imports

    def _walk_for_imports(self, node, imports, lang):
        if lang == "python":
            if node.type == "import_statement":
                for child in node.children:
                    if child.type == "dotted_name":
                        imports.append(child.text.decode())
            elif node.type == "import_from_statement":
                module = node.child_by_field_name("module_name")
                if module:
                    imports.append(f"from {module.text.decode()}")
        elif lang == "javascript":
            if node.type == "import_statement":
                imports.append(node.text.decode().split("\n")[0])

        for child in node.children:
            self._walk_for_imports(child, imports, lang)

    def _extract_python_params(self, params_node):
        """Extract parameter names from Python function params."""
        params = []
        if not params_node:
            return params
        for child in params_node.children:
            if child.type == "identifier":
                params.append(child.text.decode())
            elif child.type == "typed_parameter":
                name = child.child_by_field_name("name") if hasattr(child, "child_by_field_name") else None
                if name:
                    params.append(name.text.decode())
                else:
                    # fallback
                    for sub in child.children:
                        if sub.type == "identifier":
                            params.append(sub.text.decode())
                            break
            elif child.type == "default_parameter":
                name = child.child_by_field_name("name")
                if name:
                    params.append(name.text.decode())
        return params

    def _extract_js_params(self, params_node):
        params = []
        if not params_node:
            return params
        for child in params_node.children:
            if child.type == "identifier":
                params.append(child.text.decode())
        return params

    def _calc_nesting_depth(self, node, current=0):
        """Calculate max nesting depth in a subtree."""
        max_depth = current
        nesting_types = {
            "if_statement", "for_statement", "while_statement",
            "try_statement", "with_statement", "elif_clause",
            "else_clause", "except_clause",
        }
        for child in node.children:
            if child.type in nesting_types:
                child_depth = self._calc_nesting_depth(child, current + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calc_nesting_depth(child, current)
                max_depth = max(max_depth, child_depth)
        return max_depth

    def _is_method(self, node):
        """Check if a function is inside a class body."""
        parent = node.parent
        while parent:
            if parent.type == "class_definition":
                return True
            parent = parent.parent
        return False

    def _extract_decorators(self, node):
        """Extract decorator names from a function."""
        decorators = []
        if node.prev_sibling and node.prev_sibling.type == "decorator":
            decorators.append(node.prev_sibling.text.decode())
        return decorators

    def analyze_file(self, filepath):
        """Full analysis of a source file."""
        with open(filepath, "r") as f:
            code = f.read()

        lang = self.set_language(filepath)
        tree = self.parse(code, lang)

        result = {
            "file": str(filepath),
            "language": lang,
            "functions": self.extract_functions(tree, lang),
            "classes": self.extract_classes(tree, lang),
            "imports": self.extract_imports(tree, lang),
            "total_lines": code.count("\n") + 1,
        }

        # some basic stats
        funcs = result["functions"]
        if funcs:
            result["avg_function_length"] = sum(f["length"] for f in funcs) / len(funcs)
            result["max_nesting"] = max(f["nesting_depth"] for f in funcs)
        else:
            result["avg_function_length"] = 0
            result["max_nesting"] = 0

        return result

    def analyze_directory(self, dirpath):
        """Analyze all source files in a directory."""
        results = []
        for root, dirs, files in os.walk(dirpath):
            # skip common non-source dirs
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", ".venv", "venv"}]
            for f in files:
                if Path(f).suffix in (".py", ".js", ".jsx"):
                    filepath = os.path.join(root, f)
                    try:
                        result = self.analyze_file(filepath)
                        results.append(result)
                    except Exception as e:
                        print(f"Error analyzing {filepath}: {e}")
        return results


def main():
    parser = argparse.ArgumentParser(description="AST-based code analysis")
    parser.add_argument("file", nargs="?", help="File to analyze")
    parser.add_argument("--dir", help="Directory to analyze")
    parser.add_argument("--code", help="Code string to analyze")
    parser.add_argument("--lang", default="python", choices=["python", "javascript"])
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    analyzer = CodeAnalyzer()

    if args.dir:
        results = analyzer.analyze_directory(args.dir)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            total_files = len(results)
            total_funcs = sum(len(r["functions"]) for r in results)
            total_classes = sum(len(r["classes"]) for r in results)
            print(f"Analyzed {total_files} files: {total_funcs} functions, {total_classes} classes")
            for r in results:
                if r["functions"] or r["classes"]:
                    print(f"\n{r['file']} ({r['language']})")
                    for f in r["functions"]:
                        print(f"  fn {f['name']}() line {f['line']}, {f['length']} lines, depth {f['nesting_depth']}")
                    for c in r["classes"]:
                        print(f"  class {c['name']} line {c['line']}, {c['num_methods']} methods")

    elif args.file:
        result = analyzer.analyze_file(args.file)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"File: {result['file']} ({result['language']})")
            print(f"Lines: {result['total_lines']}")
            print(f"Functions: {len(result['functions'])}")
            print(f"Classes: {len(result['classes'])}")
            print(f"Imports: {result['imports']}")
            print(f"Avg func length: {result['avg_function_length']:.1f} lines")
            print(f"Max nesting: {result['max_nesting']}")
            print("\nFunctions:")
            for f in result["functions"]:
                params = ", ".join(f["params"])
                print(f"  {f['name']}({params}) line {f['line']} [{f['length']} lines]")

    elif args.code:
        tree = analyzer.parse(args.code, args.lang)
        funcs = analyzer.extract_functions(tree, args.lang)
        imports = analyzer.extract_imports(tree, args.lang)
        print(f"Functions: {len(funcs)}")
        for f in funcs:
            print(f"  {f['name']}({', '.join(f['params'])})")
        print(f"Imports: {imports}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
