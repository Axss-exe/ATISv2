"""
ATIS SDK Diagnostic — Run this in your Render shell or add to app startup
"""
import sys
import os
import subprocess

print("=" * 60)
print("ATIS SDK DIAGNOSTIC")
print("=" * 60)

# 1. Python info
print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path (sys.path):")
for p in sys.path:
    print(f"  {p}")

# 2. Check for local shadowing
print(f"\n--- Checking for local 'mistralai' shadowing ---")
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

# Check for mistralai.py
mistralai_py = os.path.join(cwd, "mistralai.py")
if os.path.exists(mistralai_py):
    print(f"  ❌ FOUND: {mistralai_py} — This shadows the installed package!")
else:
    print(f"  ✓ No mistralai.py in cwd")

# Check for mistralai/ directory
mistralai_dir = os.path.join(cwd, "mistralai")
if os.path.isdir(mistralai_dir):
    print(f"  ❌ FOUND: {mistralai_dir}/ — This shadows the installed package!")
else:
    print(f"  ✓ No mistralai/ directory in cwd")

# Check in parent directories
for parent in ["..", "../..", "../../.."]:
    p = os.path.abspath(os.path.join(cwd, parent, "mistralai.py"))
    if os.path.exists(p):
        print(f"  ❌ FOUND: {p} — This could shadow the package!")
    p = os.path.abspath(os.path.join(cwd, parent, "mistralai"))
    if os.path.isdir(p):
        print(f"  ❌ FOUND: {p}/ — This could shadow the package!")

# 3. Try importing mistralai
print(f"\n--- Import test ---")
try:
    import mistralai
    print(f"  ✓ import mistralai succeeded")
    print(f"    Package location: {mistralai.__file__}")
    print(f"    Package version: {getattr(mistralai, '__version__', 'unknown')}")

    try:
        from mistralai import Mistral
        print(f"  ✓ from mistralai import Mistral succeeded")
    except ImportError as e:
        print(f"  ❌ from mistralai import Mistral FAILED: {e}")
        print(f"    Available attributes: {dir(mistralai)[:20]}...")

except ImportError as e:
    print(f"  ❌ import mistralai FAILED: {e}")

# 4. Check pip list
print(f"\n--- pip list (mistralai) ---")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "mistralai"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"  pip show mistralai failed: {result.stderr}")
except Exception as e:
    print(f"  Could not run pip: {e}")

# 5. Check site-packages
print(f"\n--- site-packages contents ---")
import site
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    if sp and os.path.isdir(sp):
        mistral_pkg = os.path.join(sp, "mistralai")
        if os.path.exists(mistral_pkg):
            print(f"  ✓ Found mistralai in: {mistral_pkg}")
        mistral_egg = os.path.join(sp, "mistralai-" + os.sep)
        for item in os.listdir(sp) if os.path.isdir(sp) else []:
            if item.startswith("mistralai"):
                print(f"  ✓ Found: {os.path.join(sp, item)}")

# 6. Environment variables
print(f"\n--- Environment ---")
print(f"  LLM_PROVIDER: {os.getenv('LLM_PROVIDER', '<not set>')}")
print(f"  LLM_MODEL: {os.getenv('LLM_MODEL', '<not set>')}")
print(f"  MISTRAL_API_KEY set: {'MISTRAL_API_KEY' in os.environ}")
print(f"  LLM_API_KEY set: {'LLM_API_KEY' in os.environ}")

print(f"\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
