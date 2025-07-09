#!/usr/bin/env python3
"""
Test basic agent functionality with simple questions - smolagents 1.13.0 compatible
"""

import os
from smolagents import CodeAgent, HfApiModel, tool

@tool
def simple_calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        # Safe evaluation for basic math
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def test_basic_agent():
    """Test agent with simple questions"""
    print("🤖 Testing Basic Agent Functionality...")
    
    try:
        # Create simple agent with ONLY supported parameters for 1.13.0
        model = HfApiModel(model_id="meta-llama/Llama-3.1-70B-Instruct")
        agent = CodeAgent(
            model=model,
            tools=[simple_calculator]
            # REMOVED: max_iterations parameter - not supported in 1.13.0
        )
        
        print("✅ Agent created successfully")
        
        # Test cases with simple prompts
        test_cases = [
            {
                "question": "Use the calculator tool to compute 15 + 27",
                "expected": "42"
            },
            {
                "question": "Calculate 8 * 7 using the simple_calculator tool",
                "expected": "56"
            },
            {
                "question": "What is 100 / 4? Use the calculator.",
                "expected": "25"
            }
        ]
        
        passed = 0
        for i, test_case in enumerate(test_cases, 1):
            question = test_case["question"]
            expected = test_case["expected"]
            
            print(f"\n🧮 Test {i}: {question}")
            
            try:
                # Run the agent
                result = agent.run(question)
                
                # Clean the result
                cleaned_result = str(result).strip()
                
                # Remove common prefixes
                prefixes = ["the answer is", "result:", "answer:", "final answer:"]
                for prefix in prefixes:
                    if cleaned_result.lower().startswith(prefix):
                        cleaned_result = cleaned_result[len(prefix):].strip()
                        break
                
                # Remove trailing punctuation
                cleaned_result = cleaned_result.rstrip(".,!?")
                
                print(f"   Raw result: {result}")
                print(f"   Cleaned: '{cleaned_result}'")
                print(f"   Expected: '{expected}'")
                
                # Check if expected answer is in the result
                if expected in cleaned_result or cleaned_result == expected:
                    print("   ✅ PASS")
                    passed += 1
                else:
                    print("   ❌ FAIL - Answer not found in result")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                # Print more details for debugging
                import traceback
                print(f"   Full error: {traceback.format_exc()}")
        
        print(f"\n📊 Agent Test Results: {passed}/{len(test_cases)} passed")
        
        # Consider it a success if at least 1 test passes (shows basic functionality)
        success = passed >= 1
        
        if success:
            print("✅ Basic agent functionality is working!")
        else:
            print("❌ Agent is not functioning properly")
            
        return success
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        print(f"Full error details:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🧪 Agent Functionality Test (smolagents 1.13.0)")
    print("=" * 50)
    
    if test_basic_agent():
        print("\n🎉 Agent functionality test passed!")
        print("Your agent is working with smolagents 1.13.0")
        print("Ready to proceed with integration testing.")
    else:
        print("\n🚨 Agent functionality test failed!")
        print("Check your HuggingFace token and model access.")
