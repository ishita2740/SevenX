import json, math, sys, statistics

data = json.loads(sys.stdin.read())
expr = data["params"].get("expression", "")

safe_names = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
    "int": int, "float": float,
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    "sqrt": math.sqrt, "cbrt": lambda x: x ** (1/3),
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "pow": math.pow,
    "ceil": math.ceil, "floor": math.floor,
    "factorial": math.factorial, "gcd": math.gcd,
    "radians": math.radians, "degrees": math.degrees,
    "mean": lambda *a: statistics.mean(a),
    "median": lambda *a: statistics.median(a),
    "stdev": lambda *a: statistics.stdev(a) if len(a) > 1 else 0,
}

try:
    if not expr.strip():
        print(json.dumps({"success": False, "error": "Empty expression"}))
    else:
        result = eval(expr, {"__builtins__": {}}, safe_names)
        print(json.dumps({"success": True, "data": {
            "expression": expr,
            "result": result,
            "type": type(result).__name__,
        }}))
except Exception as e:
    print(json.dumps({"success": False, "error": f"Calc error: {e}"}))
