#!/usr/bin/env python3
"""
Test integration with your existing app.py structure
"""

def test_app_integration():
    """Test that your app.py can import and use the agent"""
    print("🔧 Testing app.py Integration...")
    
    try:
        # Test importing your simple agent
        from agents import SimpleGAIAAgent
        print("✅ SimpleGAIAAgent imported successfully")
        
        # Test BasicAgent integration
        class BasicAgent:
            def __init__(self):
                self.agent = SimpleGAIAAgent()
            
            def __call__(self, question: str) -> str:
                return self.agent.solve(question)
        
        # Test basic functionality
        agent = BasicAgent()
        result = agent("What is 2 + 2?")
        
        print(f"✅ Integration test result: {result}")
        
        # Test answer cleaning
        if "4" in str(result):
            print("✅ Basic math working correctly")
            return True
        else:
            print("❌ Unexpected result from integration test")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have created the simple_agent.py file")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Integration Test")
    print("=" * 30)
    
    if test_app_integration():
        print("\n🎉 Integration test passed!")
        print("Ready to run your full app.py!")
    else:
        print("\n🚨 Integration test failed!")
        print("Fix integration issues before running app.py")
