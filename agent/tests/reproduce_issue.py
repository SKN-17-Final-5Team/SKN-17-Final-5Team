import sys
import os
import json
from unittest.mock import MagicMock

# Mock tavily to avoid import error
sys.modules["tavily"] = MagicMock()

# Add agent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.document_generation_tool import generate_trade_document

def test_stateful_generation():
    print("--- Starting Stateful Generation Test ---")
    
    # Unwrap the function_tool
    print(f"Type of generate_trade_document: {type(generate_trade_document)}")
    print(f"Dir: {dir(generate_trade_document)}")
    try:
        print(f"Vars: {vars(generate_trade_document)}")
    except:
        pass
    
    if hasattr(generate_trade_document, 'func'):
        target_func = generate_trade_document.func
        print("✓ unwrapped .func")
    elif hasattr(generate_trade_document, '__wrapped__'):
        target_func = generate_trade_document.__wrapped__
        print("✓ unwrapped .__wrapped__")
    else:
        target_func = generate_trade_document
        print("! using original object")

    # 1. Initial Generation
    print("\n1. Generating Offer Sheet with Seller Name...")
    data1 = json.dumps({"Seller Name": "Old Corp"})
    result1 = target_func("Offer_Sheet", data1)
    
    if "Old Corp" in result1:
        print("✓ Initial generation successful (contains 'Old Corp')")
    else:
        print("✗ Initial generation failed")

    # 2. Modification (Adding Buyer Name)
    print("\n2. Modifying Offer Sheet (Adding Buyer Name)...")
    data2 = json.dumps({"Buyer Name": "New Buyer"})
    result2 = target_func("Offer_Sheet", data2)
    
    # Check if BOTH exist
    has_old = "Old Corp" in result2
    has_new = "New Buyer" in result2
    
    if has_old and has_new:
        print("✓ State persistence successful (contains both 'Old Corp' and 'New Buyer')")
    elif has_new and not has_old:
        print("✗ State persistence FAILED (Old data lost)")
    else:
        print(f"✗ Unexpected result: {result2[:100]}...")

    # 3. Overwrite Test
    print("\n3. Testing Overwrite (New Item)...")
    data3 = json.dumps({"Item No": "999"})
    result3 = target_func("Offer_Sheet", data3, overwrite=True)
    
    has_old_in_3 = "Old Corp" in result3
    has_new_in_3 = "New Buyer" in result3
    has_item_in_3 = "999" in result3
    
    if has_item_in_3 and not has_old_in_3 and not has_new_in_3:
        print("✓ Overwrite successful (Only new data exists)")
    else:
        print(f"✗ Overwrite failed. Old: {has_old_in_3}, New: {has_new_in_3}, Item: {has_item_in_3}")

if __name__ == "__main__":
    test_stateful_generation()
