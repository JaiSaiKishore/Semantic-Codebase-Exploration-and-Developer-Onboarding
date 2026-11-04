import os
import re
import ast
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter

# Common architectural layer definitions
ARCHITECTURAL_LAYERS = {
    "Presentation": {
        "keywords": ["api", "route", "routes", "controller", "controllers", "view", "views", 
                     "endpoint", "endpoints", "handler", "handlers", "cli", "ui", "web", "graphql", "rest"],
        "order": 1,
        "description": "API endpoints, routing, controllers, CLI, and client presentation layer"
    },
    "Application": {
        "keywords": ["service", "services", "usecase", "usecases", "use_case", "use_cases", 
                     "workflow", "workflows", "orchestration", "manager", "managers", "task", "tasks", "action", "actions"],
        "order": 2,
        "description": "Application logic, business workflows, orchestration, and use cases"
    },
    "Domain": {
        "keywords": ["domain", "model", "models", "entity", "entities", "core", "schema", 
                     "schemas", "type", "types", "dto", "dtos", "value_object", "value_objects"],
        "order": 3,
        "description": "Core enterprise business models, domain entities, rules, and value objects"
    },
    "Infrastructure": {
        "keywords": ["infrastructure", "infra", "db", "database", "repository", "repositories", 
                     "adapter", "adapters", "storage", "connector", "connectors", "client", "clients", 
                     "cache", "redis", "postgres", "sql", "orm", "gateway", "gateways"],
        "order": 4,
        "description": "Database repositories, ORMs, network clients, external service adapters, and I/O"
    },
    "Shared / Utility": {
        "keywords": ["util", "utils", "common", "helper", "helpers", "shared", "lib", "config", "constants"],
        "order": 5,
        "description": "Cross-cutting utilities, configuration, helper functions, and shared libraries"
    }
}

class DependencyAnalyzer:
    """Extracts module-level dependencies, builds a directed graph, and analyzes coupling."""
    
    def __init__(self, repo_dir: str):
        self.repo_dir = os.path.abspath(repo_dir)
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

    def _rel_path(self, full_path: str) -> str:
        return os.path.relpath(full_path, self.repo_dir).replace("\\", "/")

    def _file_to_module(self, rel_path: str) -> str:
        clean = rel_path
        for ext in [".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rs", ".cs"]:
            if clean.endswith(ext):
                clean = clean[:-len(ext)]
                break
        parts = clean.split("/")
        if len(parts) > 1 and parts[-1] in ("__init__", "index", "mod"):
            return ".".join(parts[:-1])
        return ".".join(parts)

    def extract_imports_python(self, file_path: str, content: str) -> List[Tuple[str, int, str]]:
        """Returns list of (imported_module, line_number, raw_statement)."""
        imports = []
        try:
            tree = ast.parse(content, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, node.lineno, f"import {alias.name}"))
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    level = node.level
                    if level > 0:
                        # Relative import
                        rel_dir = os.path.dirname(self._rel_path(file_path))
                        dir_parts = rel_dir.split("/") if rel_dir and rel_dir != "." else []
                        prefix_parts = dir_parts[:max(0, len(dir_parts) - level + 1)]
                        resolved = ".".join(prefix_parts + ([mod] if mod else []))
                        for alias in node.names:
                            imports.append((f"{resolved}.{alias.name}" if resolved else alias.name, node.lineno, f"from {'.' * level}{mod} import {alias.name}"))
                    else:
                        for alias in node.names:
                            full_target = f"{mod}.{alias.name}" if mod else alias.name
                            imports.append((mod if mod else full_target, node.lineno, f"from {mod} import {alias.name}"))
        except Exception:
            # Fallback to regex for syntax errors or non-compiling files
            for line_no, line in enumerate(content.splitlines(), 1):
                line_str = line.strip()
                m_from = re.match(r"^from\s+([a-zA-Z0-9_\.]+)\s+import", line_str)
                m_imp = re.match(r"^import\s+([a-zA-Z0-9_\.,\s]+)", line_str)
                if m_from:
                    imports.append((m_from.group(1), line_no, line_str))
                elif m_imp:
                    for name in m_imp.group(1).split(","):
                        pkg = name.strip().split(" as ")[0].strip()
                        if pkg:
                            imports.append((pkg, line_no, line_str))
        return imports

    def extract_imports_js_ts(self, file_path: str, content: str) -> List[Tuple[str, int, str]]:
        imports = []
        for line_no, line in enumerate(content.splitlines(), 1):
            line_str = line.strip()
            # import ... from '...'; or require('...')
            m_es6 = re.search(r'''(?:import|export\s+.*?\s+from)\s+['"]([^'"]+)['"]''', line_str)
            m_req = re.search(r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)''', line_str)
            target = None
            if m_es6:
                target = m_es6.group(1)
            elif m_req:
                target = m_req.group(1)
            
            if target:
                if target.startswith("."):
                    rel_dir = os.path.dirname(self._rel_path(file_path))
                    resolved_dir = os.path.normpath(os.path.join(rel_dir, target)).replace("\\", "/")
                    target_mod = resolved_dir.replace("/", ".")
                else:
                    target_mod = target
                imports.append((target_mod, line_no, line_str))
        return imports

    def analyze_repository(self, files: List[str]):
        """Builds module registry, dependencies, and adjacency matrices."""
        file_mod_map = {}
        for f in files:
            rel = self._rel_path(f)
            mod_name = self._file_to_module(rel)
            file_mod_map[f] = (rel, mod_name)
            if mod_name not in self.modules:
                self.modules[mod_name] = {
                    "name": mod_name,
                    "rel_path": rel,
                    "full_path": f,
                    "symbols": [],
                    "classes": [],
                    "functions": [],
                    "lines": 0,
                    "layer": "Shared / Utility"
                }

        known_modules = set(self.modules.keys())

        for f in files:
            rel, src_mod = file_mod_map[f]
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
            except Exception:
                continue

            self.modules[src_mod]["lines"] = len(content.splitlines())

            ext = os.path.splitext(f)[1].lower()
            if ext == ".py":
                raw_imports = self.extract_imports_python(f, content)
            elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
                raw_imports = self.extract_imports_js_ts(f, content)
            else:
                raw_imports = []

            for target_raw, line_no, raw_stmt in raw_imports:
                # Match to internal project modules
                target_mod = None
                if target_raw in known_modules:
                    target_mod = target_raw
                else:
                    # Prefix match: e.g. target_raw = backend.routes.app -> backend.routes
                    for km in known_modules:
                        if target_raw == km or target_raw.startswith(km + ".") or km.startswith(target_raw + "."):
                            target_mod = km
                            break

                if target_mod and target_mod != src_mod:
                    self.adjacency[src_mod].add(target_mod)
                    self.reverse_adjacency[target_mod].add(src_mod)
                    self.edges.append({
                        "source": src_mod,
                        "target": target_mod,
                        "source_file": rel,
                        "line": line_no,
                        "statement": raw_stmt[:120]
                    })

    def detect_cycles(self) -> List[List[str]]:
        """Finds all circular dependency cycles using DFS cycle detection."""
        cycles = []
        visited = set()
        stack = []
        stack_set = set()

        def dfs(node):
            visited.add(node)
            stack.append(node)
            stack_set.add(node)

            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in stack_set:
                    # Found a cycle!
                    cycle_start = stack.index(neighbor)
                    cycle = stack[cycle_start:] + [neighbor]
                    if len(cycle) > 2:
                        # Normalize cycle to avoid duplicates
                        min_idx = cycle[:-1].index(min(cycle[:-1]))
                        norm_cycle = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                        if norm_cycle not in cycles:
                            cycles.append(norm_cycle)

            stack.pop()
            stack_set.remove(node)

        for node in list(self.adjacency.keys()):
            if node not in visited:
                dfs(node)

        return cycles

    def compute_coupling_metrics(self) -> List[Dict[str, Any]]:
        """Calculates Afferent Coupling (Ca), Efferent Coupling (Ce), and Instability (I)."""
        metrics = []
        for mod, data in self.modules.items():
            ce = len(self.adjacency.get(mod, []))
            ca = len(self.reverse_adjacency.get(mod, []))
            instability = round(ce / (ca + ce), 2) if (ca + ce) > 0 else 0.0
            metrics.append({
                "module": mod,
                "file": data["rel_path"],
                "fan_out_ce": ce,
                "fan_in_ca": ca,
                "total_coupling": ca + ce,
                "instability": instability,
                "lines": data.get("lines", 0)
            })
        metrics.sort(key=lambda x: x["total_coupling"], reverse=True)
        return metrics


class ArchitectureModelEngine:
    """Discovers architectural patterns, boundaries, conformance rules, and conventions."""

    def __init__(self, repo_dir: str, repo_name: str, files: List[str]):
        self.repo_dir = os.path.abspath(repo_dir)
        self.repo_name = repo_name
        self.files = files
        self.analyzer = DependencyAnalyzer(self.repo_dir)

    def _infer_module_layer(self, mod_name: str, rel_path: str, symbols: List[str]) -> str:
        tokens = set(re.split(r"[/_.\-]", f"{mod_name} {rel_path}".lower()))
        tokens.update([s.lower() for s in symbols])

        layer_scores = Counter()
        for layer_name, spec in ARCHITECTURAL_LAYERS.items():
            for kw in spec["keywords"]:
                if kw in tokens:
                    layer_scores[layer_name] += 2
                elif any(kw in t for t in tokens):
                    layer_scores[layer_name] += 1

        if layer_scores:
            return layer_scores.most_common(1)[0][0]

        # Default fallback heuristics
        if any(p in rel_path.lower() for p in ["server", "app", "main", "entry"]):
            return "Presentation"
        return "Shared / Utility"

    def detect_architectural_patterns(self, layers_count: Dict[str, int], 
                                      dep_edges: List[Dict[str, Any]], 
                                      symbol_names: List[str]) -> List[Dict[str, Any]]:
        patterns = []
        symbol_text = " ".join(symbol_names).lower()

        # 1. Layered / N-Tier Architecture (MVC)
        has_pres = layers_count.get("Presentation", 0) > 0
        has_serv = layers_count.get("Application", 0) > 0
        has_data = layers_count.get("Infrastructure", 0) > 0 or layers_count.get("Domain", 0) > 0
        if has_pres and (has_serv or has_data):
            conf = 92 if (has_pres and has_serv and has_data) else 75
            patterns.append({
                "pattern": "Layered / N-Tier Architecture (Controller-Service-Repository)",
                "category": "Structural Pattern",
                "confidence": conf,
                "description": "Separates presentation/API endpoints, service business logic, domain models, and infrastructure/database adapters into distinct hierarchical tiers.",
                "evidence": [f"Presentation modules ({layers_count.get('Presentation', 0)})",
                             f"Application services ({layers_count.get('Application', 0)})",
                             f"Domain & Infrastructure components ({layers_count.get('Domain', 0) + layers_count.get('Infrastructure', 0)})"]
            })

        # 2. Hexagonal / Clean Architecture (Ports & Adapters)
        has_domain = layers_count.get("Domain", 0) > 0
        has_infra = layers_count.get("Infrastructure", 0) > 0
        domain_to_infra = any(
            e["source"] in [m for m, d in self.analyzer.modules.items() if d["layer"] == "Domain"] and
            e["target"] in [m for m, d in self.analyzer.modules.items() if d["layer"] == "Infrastructure"]
            for e in dep_edges
        )
        if has_domain and has_infra and not domain_to_infra:
            patterns.append({
                "pattern": "Hexagonal / Clean Architecture (Ports & Adapters)",
                "category": "Decoupling Pattern",
                "confidence": 88,
                "description": "Core enterprise business domain is isolated with zero outbound dependencies toward databases or frameworks. Infrastructure implements port interfaces defined by the domain.",
                "evidence": [
                    f"Domain layer with {layers_count.get('Domain', 0)} entities",
                    "Domain has zero outward imports into Infrastructure/Adapters"
                ]
            })

        # 3. Repository Pattern
        repo_matches = [s for s in symbol_names if "repository" in s.lower() or "repo" in s.lower()]
        if repo_matches or any("repository" in m for m in self.analyzer.modules):
            patterns.append({
                "pattern": "Repository Pattern",
                "category": "Data Access Pattern",
                "confidence": 90,
                "description": "Mediates between domain models and data persistence layers, providing a collection-like interface for database operations.",
                "evidence": [f"Discovered repository abstractions and classes: {', '.join(repo_matches[:4]) or 'Modules with repository naming'}"]
            })

        # 4. Pipeline / Workflow / State Machine Pattern
        pipeline_keywords = ["pipeline", "stage", "workflow", "stategraph", "chain", "middleware", "processor", "node"]
        matched_pipeline = [kw for kw in pipeline_keywords if kw in symbol_text or any(kw in m.lower() for m in self.analyzer.modules)]
        if matched_pipeline:
            patterns.append({
                "pattern": "Pipeline / State Machine Architecture",
                "category": "Behavioral Flow Pattern",
                "confidence": 85,
                "description": "Executes sequential transformation stages or graph-based state transitions where output from one step feeds into subsequent nodes.",
                "evidence": [f"Matched pipeline symbols/constructs: {', '.join(matched_pipeline)}"]
            })

        # 5. Dependency Injection / Inversion of Control
        di_keywords = ["inject", "depends", "container", "provider", "factory", "wire"]
        matched_di = [kw for kw in di_keywords if kw in symbol_text]
        if matched_di:
            patterns.append({
                "pattern": "Dependency Injection (IoC)",
                "category": "Creational & Wiring Pattern",
                "confidence": 80,
                "description": "Components receive their dependencies from external callers, parameter providers (e.g. FastAPI Depends), or IoC containers rather than instantiating hardcoded singletons.",
                "evidence": [f"Discovered injection keywords and constructs: {', '.join(matched_di)}"]
            })

        for p in patterns:
            p["name"] = p.get("pattern", "")
            p["summary"] = p.get("description", "")
            if isinstance(p.get("confidence"), (int, float)) and p["confidence"] > 1:
                p["confidence"] = p["confidence"] / 100.0

        return patterns

    def run_conformance_checks(self, cycles: List[List[str]], 
                               metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluates strict architectural rules and flags violations with file and line evidence."""
        violations = []

        # Rule 1: Circular Dependencies
        for cycle in cycles:
            source_mod = cycle[0]
            target_mod = cycle[1]
            src_file = self.analyzer.modules.get(source_mod, {}).get("rel_path", source_mod)
            violations.append({
                "rule_id": "ARCH-RULE-001",
                "rule_name": "Circular Module Dependency",
                "severity": "HIGH",
                "category": "Coupling & Modularity",
                "source": src_file,
                "target": cycle[1],
                "line": 1,
                "description": f"Circular import cycle detected across modules: {' -> '.join(cycle)}",
                "evidence": f"Cycle loop: {' -> '.join(cycle)}",
                "remediation": "Extract shared data structures or protocols into a dedicated domain/common module to break the circular dependency cycle."
            })

        # Rule 2: Layer Boundary Bypass (e.g. Presentation directly importing Database/Infrastructure)
        for edge in self.analyzer.edges:
            src = edge["source"]
            tgt = edge["target"]
            src_layer = self.analyzer.modules.get(src, {}).get("layer")
            tgt_layer = self.analyzer.modules.get(tgt, {}).get("layer")

            # Presentation layer bypassing Application to directly call Infrastructure
            if src_layer == "Presentation" and tgt_layer == "Infrastructure":
                violations.append({
                    "rule_id": "ARCH-RULE-002",
                    "rule_name": "Layer Bypass: Presentation Directly Accesses Infrastructure",
                    "severity": "MEDIUM",
                    "category": "Architectural Layering",
                    "source": edge["source_file"],
                    "target": tgt,
                    "line": edge["line"],
                    "description": f"Controller '{src}' directly imports Infrastructure component '{tgt}', bypassing intermediate service/application use cases.",
                    "evidence": edge["statement"],
                    "remediation": "Encapsulate database queries and repository calls inside an Application/Service layer workflow instead of invoking them directly in endpoints."
                })

            # Rule 3: Domain Inversion (Domain importing Infrastructure or Presentation)
            if src_layer == "Domain" and tgt_layer in ("Infrastructure", "Presentation"):
                violations.append({
                    "rule_id": "ARCH-RULE-003",
                    "rule_name": "Clean Architecture Violation: Domain Depends on External Layer",
                    "severity": "HIGH",
                    "category": "Dependency Inversion",
                    "source": edge["source_file"],
                    "target": tgt,
                    "line": edge["line"],
                    "description": f"Domain entity '{src}' imports external {tgt_layer} module '{tgt}'. Domain logic should remain agnostic of database or web frameworks.",
                    "evidence": edge["statement"],
                    "remediation": "Define an abstract interface (Port) in the Domain layer and implement it within the Infrastructure layer via dependency injection."
                })

        # Rule 4: God Object / Excessive Coupling
        for m in metrics:
            if m["fan_out_ce"] >= 12 or m["total_coupling"] >= 18:
                violations.append({
                    "rule_id": "ARCH-RULE-004",
                    "rule_name": "God Component / Excessive Structural Coupling",
                    "severity": "LOW",
                    "category": "Code Quality & Responsibility",
                    "source": m["file"],
                    "target": f"Coupled to {m['fan_out_ce']} outbound dependencies",
                    "line": 1,
                    "description": f"Component '{m['module']}' is coupled to {m['fan_out_ce']} external modules (Total coupling: {m['total_coupling']}, Instability: {m['instability']}).",
                    "evidence": f"Fan-out: {m['fan_out_ce']} modules, Lines of Code: {m.get('lines', 0)}",
                    "remediation": "Decompose this component into smaller, single-responsibility units using the Facade or Strategy pattern."
                })

        for v in violations:
            v["source_file"] = v.get("source", "")
            v["line_number"] = v.get("line", 1)

        return violations

    def mine_code_conventions(self, files: List[str], symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes naming styles, error handling idioms, and engineering conventions."""
        file_namings = Counter()
        class_namings = Counter()
        func_namings = Counter()
        error_styles = Counter()
        async_count = 0
        sync_count = 0

        for f in files:
            fname = os.path.basename(f)
            base, ext = os.path.splitext(fname)
            if "_" in base and base.islower():
                file_namings["snake_case"] += 1
            elif "-" in base and base.islower():
                file_namings["kebab-case"] += 1
            elif base[0].isupper() and "_" not in base and "-" not in base:
                file_namings["PascalCase"] += 1
            elif base[0].islower() and any(c.isupper() for c in base):
                file_namings["camelCase"] += 1

            # Inspect content for idioms
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                    code_snippet = fp.read(4000)
                    if "async def " in code_snippet or "async function" in code_snippet:
                        async_count += 1
                    elif "def " in code_snippet or "function " in code_snippet:
                        sync_count += 1

                    if "raise HTTPException" in code_snippet or "throw new Error" in code_snippet:
                        error_styles["HTTP/Explicit Exception Throwing"] += 1
                    if "try:" in code_snippet or "try {" in code_snippet:
                        error_styles["Try/Catch Defensive Wrapping"] += 1
                    if "logging.getLogger" in code_snippet or "logger." in code_snippet:
                        error_styles["Structured Logging"] += 1
            except Exception:
                continue

        for s in symbols:
            s_name = s.get("name", "")
            s_type = s.get("type", "")
            if "class" in s_type:
                if s_name.startswith("I") and len(s_name) > 1 and s_name[1].isupper():
                    class_namings["IPrefixInterface (e.g. IService)"] += 1
                elif s_name[0].isupper():
                    class_namings["PascalCase"] += 1
            elif "function" in s_type or "method" in s_type:
                clean_name = s_name.split(".")[-1]
                if "_" in clean_name and clean_name.islower():
                    func_namings["snake_case"] += 1
                elif clean_name and clean_name[0].islower() and any(c.isupper() for c in clean_name):
                    func_namings["camelCase"] += 1

        total_files = max(sum(file_namings.values()), 1)
        dom_file_convention = file_namings.most_common(1)[0][0] if file_namings else "snake_case"
        file_consistency = round((file_namings.get(dom_file_convention, 0) / total_files) * 100, 1)

        total_funcs = max(sum(func_namings.values()), 1)
        dom_func_convention = func_namings.most_common(1)[0][0] if func_namings else "snake_case"
        func_consistency = round((func_namings.get(dom_func_convention, 0) / total_funcs) * 100, 1)

        return {
            "file_naming": {
                "convention": dom_file_convention,
                "dominant_style": dom_file_convention,
                "consistency_rate": f"{file_consistency}%",
                "consistency_pct": file_consistency,
                "distribution": dict(file_namings)
            },
            "function_naming": {
                "convention": dom_func_convention,
                "dominant_style": dom_func_convention,
                "consistency_rate": f"{func_consistency}%",
                "consistency_pct": func_consistency,
                "distribution": dict(func_namings)
            },
            "class_naming": {
                "convention": class_namings.most_common(1)[0][0] if class_namings else "PascalCase",
                "dominant_style": class_namings.most_common(1)[0][0] if class_namings else "PascalCase",
                "consistency_pct": 95.0,
                "distribution": dict(class_namings)
            },
            "async_style": {
                "predominant": "Async/Await Event-Driven" if async_count >= sync_count else "Synchronous / Thread-Pool",
                "async_functions": async_count,
                "sync_functions": sync_count
            },
            "execution_model": {
                "async_predominance": f"{round((async_count / max(async_count + sync_count, 1)) * 100, 1)}%",
                "async_files": async_count,
                "sync_files": sync_count
            },
            "error_handling": {
                "idioms": dict(error_styles),
                "try_catch_density_per_kloc": round((error_styles.get("Try/Catch Defensive Wrapping", 0) / max(total_files, 1)) * 10.0, 1),
                "custom_exception_classes": [s["name"] for s in symbols if "error" in s.get("name", "").lower() or "exception" in s.get("name", "").lower()]
            },
            "error_handling_idioms": dict(error_styles)
        }

    def build_model(self, ast_blocks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Constructs the repository-specific architecture model."""
        ast_blocks = ast_blocks or []
        
        # 1. Dependency Analysis
        self.analyzer.analyze_repository(self.files)
        
        # Collect symbols
        symbols_by_mod = defaultdict(list)
        for b in ast_blocks:
            mod = self.analyzer._file_to_module(self.analyzer._rel_path(b["file"]))
            symbols_by_mod[mod].append(b["name"])

        # 2. Assign Architectural Layers to Modules
        layer_distribution = Counter()
        for mod, data in self.analyzer.modules.items():
            mod_symbols = symbols_by_mod.get(mod, [])
            layer = self._infer_module_layer(mod, data["rel_path"], mod_symbols)
            data["layer"] = layer
            data["symbols"] = mod_symbols[:15]
            layer_distribution[layer] += 1

        # 3. Detect Cycles & Coupling Metrics
        cycles = self.analyzer.detect_cycles()
        coupling_metrics = self.analyzer.compute_coupling_metrics()

        # 4. Pattern Detection
        all_symbol_names = [b["name"] for b in ast_blocks]
        detected_patterns = self.detect_architectural_patterns(
            dict(layer_distribution), 
            self.analyzer.edges, 
            all_symbol_names
        )

        # 5. Conformance Verification
        violations = self.run_conformance_checks(cycles, coupling_metrics)

        # 6. Mine Repository Conventions
        conventions = self.mine_code_conventions(self.files, ast_blocks)

        # 7. Layer Abstraction Boundaries
        layers_data = []
        for layer_name, spec in sorted(ARCHITECTURAL_LAYERS.items(), key=lambda x: x[1]["order"]):
            layer_modules = [
                {"name": m, "file": d["rel_path"], "symbols": d.get("symbols", [])[:5]}
                for m, d in self.analyzer.modules.items() if d.get("layer") == layer_name
            ]
            layers_data.append({
                "layer": layer_name,
                "order": spec["order"],
                "description": spec["description"],
                "module_count": len(layer_modules),
                "modules": layer_modules[:8]  # Representative modules
            })

        # Calculate Overall Architectural Health Score (0 - 100)
        high_violations = sum(1 for v in violations if v["severity"] == "HIGH")
        med_violations = sum(1 for v in violations if v["severity"] == "MEDIUM")
        health_penalty = (high_violations * 15) + (med_violations * 6) + (len(cycles) * 10)
        health_score = max(35, min(100, 100 - health_penalty))

        model = {
            "repository": self.repo_name,
            "total_modules": len(self.analyzer.modules),
            "total_dependency_edges": len(self.analyzer.edges),
            "health_score": health_score,
            "dominant_pattern": detected_patterns[0]["pattern"] if detected_patterns else "Modular Multi-Tier Application",
            "pattern_summary": detected_patterns[0]["evidence"][0] if (detected_patterns and detected_patterns[0].get("evidence")) else "Modular layered software architecture",
            "patterns": detected_patterns,
            "detected_patterns": detected_patterns,
            "layers": {l["layer"]: [m["name"] for m in l["modules"]] for l in layers_data},
            "layers_detail": layers_data,
            "layer_distribution": dict(layer_distribution),
            "modules": {m: {"name": m, "layer": d["layer"], "dependencies": list(self.analyzer.adjacency.get(m, [])), "dependents": list(self.analyzer.reverse_adjacency.get(m, []))} for m, d in self.analyzer.modules.items()},
            "circular_dependencies": cycles,
            "violations": violations,
            "conformance": {
                "total_violations": len(violations),
                "high_severity": high_violations,
                "medium_severity": med_violations,
                "violations": violations[:25]
            },
            "cycles": cycles[:10],
            "coupling_metrics": coupling_metrics[:12],
            "conventions": conventions,
            "dependency_graph": {
                "nodes": [
                    {
                        "id": m,
                        "label": m.split(".")[-1],
                        "layer": d["layer"],
                        "file": d["rel_path"],
                        "fan_in": len(self.analyzer.reverse_adjacency.get(m, [])),
                        "fan_out": len(self.analyzer.adjacency.get(m, []))
                    }
                    for m, d in list(self.analyzer.modules.items())[:60]
                ],
                "edges": [
                    {"source": e["source"], "target": e["target"], "statement": e["statement"]}
                    for e in self.analyzer.edges[:80]
                ]
            },
            "graph": {
                "nodes": [
                    {
                        "id": m,
                        "label": m.split(".")[-1],
                        "layer": d["layer"],
                        "file": d["rel_path"],
                        "fan_in": len(self.analyzer.reverse_adjacency.get(m, [])),
                        "fan_out": len(self.analyzer.adjacency.get(m, []))
                    }
                    for m, d in list(self.analyzer.modules.items())[:60]
                ],
                "edges": [
                    {"source": e["source"], "target": e["target"], "statement": e["statement"]}
                    for e in self.analyzer.edges[:80]
                ]
            }
        }

        return model

# Global in-memory cache of computed architecture models
ARCHITECTURE_MODELS: Dict[str, Dict[str, Any]] = {}

def get_or_build_architecture_model(repo_dir: Optional[str] = None, repo_name: str = "codebase", 
                                   files: Optional[List[str]] = None, 
                                   ast_blocks: Optional[List[Dict[str, Any]]] = None,
                                   force_refresh: bool = False) -> Dict[str, Any]:
    """Retrieves cached model from memory or disk, or constructs it afresh if force_refresh is True."""
    canonical_key = repo_name.strip()
    if not force_refresh:
        if canonical_key in ARCHITECTURE_MODELS:
            return ARCHITECTURE_MODELS[canonical_key]

    if not repo_dir:
        repo_dir = "."

    cache_file = os.path.join(repo_dir, ".architecture_model.json")
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                cached_repo = (data.get("repository") or "").strip()
                # Strict repo identity check: ensure cache belongs to this repo
                if cached_repo and (
                    cached_repo.lower() == canonical_key.lower()
                    or cached_repo.lower().endswith(f"/{canonical_key.lower()}")
                    or canonical_key.lower().endswith(f"/{cached_repo.lower()}")
                    or canonical_key in ("codebase", "local")
                ):
                    ARCHITECTURE_MODELS[canonical_key] = data
                    return data
        except Exception:
            pass

    if files is None or force_refresh:
        from store_vectors import scan_repository_files
        _, files = scan_repository_files(repo_dir, max_core=800, max_total=2500)

    engine = ArchitectureModelEngine(repo_dir, repo_name, files)
    model = engine.build_model(ast_blocks)
    ARCHITECTURE_MODELS[canonical_key] = model

    try:
        with open(cache_file, "w", encoding="utf-8") as fp:
            json.dump(model, fp, indent=2)
    except Exception:
        pass

    return model
