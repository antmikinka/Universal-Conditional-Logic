#!/usr/bin/env python3
"""Simple test to verify imports work"""

import sys
import os

# Add parent directory to path to import UCL modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

try:
    import core
    print("✅ core module loaded")
except ImportError as e:
    print(f"❌ core import failed: {e}")
    exit(1)

try:
    import domains
    print("✅ domains module loaded")
except ImportError as e:
    print(f"❌ domains import failed: {e}")
    exit(1)

try:
    import builder
    print("✅ builder module loaded")
except ImportError as e:
    print(f"❌ builder import failed: {e}")
    exit(1)

print("\n✅ ALL MODULES LOADED SUCCESSFULLY!")
print("\nTesting basic UCL builder...")

from builder import UCLPromptBuilder, JSONOutput
from domains import LinearAlgebraDomain

try:
    prompt = (UCLPromptBuilder()
        .set_identity("Test")
        .add_critical()
        .set_output(JSONOutput(fields=["result"]))
        .build()
    )
    print(f"✅ Built prompt - Quality: {prompt.score.Q_final:.1%}")
except Exception as e:
    print(f"❌ Build failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🎉 UCL Framework is working correctly!")
