#!/usr/bin/env python3
"""
Basic installation test for smolagents 1.13.0 and GAIA agent
"""

import os
import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import smolagents
        print(f"✅ smolagents version: {smolagents.__version__}")
    except ImportError as e:
        print(f"❌ smolagents import failed: {e}")
        return False
    
    try:
        from smolagents import CodeAgent, HfApiModel
        print("✅ Core smolagents classes imported")
    except ImportError as e:
        print(f"❌ smolagents classes import failed: {e}")
        return False
    
    try:
        import requests, pandas, numpy
        print("✅ Supporting libraries imported")
    except ImportError as e:
        print(f"❌ Supporting libraries failed: {e}")
        return False
    
    return True

def test_hf_token():
    """Test HuggingFace token"""
    print("\n🔑 Testing HuggingFace token...")
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set")
        print("Set with: export HF_TOKEN='your_token_here'")
        return False
    
    print(f"✅ HF_TOKEN configured: {token[:20]}...")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=token)
        print(f"✅ Token valid for user: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Token validation failed: {e}")
        return False

def test_model_access():
    """Test basic model access"""
    print("\n🤖 Testing model access...")
    
    try:
        from smolagents import HfApiModel
        model = HfApiModel(model_id="meta-llama/Llama-3.1-70B-Instruct")
        print("✅ Model initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 smolagents 1.13.0 Installation Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("HF Token", test_hf_token), 
        ("Model Access", test_model_access)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<20} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Ready to test agent functionality.")
    else:
        print("\n🚨 Some tests failed. Fix issues before proceeding.")
    
    sys.exit(0 if all_passed else 1)
