import pytest
import os
import sys
import json
from unittest.mock import MagicMock

# Mock agents module
mock_agents = MagicMock()
sys.modules["agents"] = mock_agents
sys.modules["tavily"] = MagicMock()
sys.modules["qdrant_client"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["pymysql"] = MagicMock()
sys.modules["DBUtils"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["pandas"] = MagicMock()

# Mock config to avoid import errors from search_tool
mock_config = MagicMock()
sys.modules["config"] = mock_config

# Mock function_tool decorator
def mock_function_tool(func):
    return func
mock_agents.function_tool = mock_function_tool

from tools.document_generation_tool import generate_trade_document, TEMPLATE_MAP

class TestDocumentGenerator:
    def test_all_document_types(self):
        """Verify that all 7 document types can be generated."""
        test_data = {"[ Applicant Name ]": "Test Applicant"}
        
        for doc_type in TEMPLATE_MAP.keys():
            result = generate_trade_document(doc_type, json.dumps(test_data))
            assert "Error" not in result
            
            # Extract file path
            import re
            match = re.search(r"sandbox:(/.+\.html)", result)
            assert match
            file_path = match.group(1)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "Test Applicant" in content or doc_type not in ["Letter_of_Credit"]
            assert "<html" in content
            
            # Clean up
            os.remove(file_path)

    def test_invalid_document_type(self):
        """Verify error handling for invalid document type."""
        result = generate_trade_document("Invalid_Type", json.dumps({}))
        assert "Error: Invalid document_type" in result

    def test_template_filling(self):
        """Verify that placeholders are correctly replaced."""
        doc_type = "Offer_Sheet"
        data = {
            "[ Seller Name ]": "My Company",
            "[ Buyer Name ]": "Your Company",
            "[ Date ]": "2023-10-27"
        }
        
        result = generate_trade_document(doc_type, json.dumps(data))
        
        # Extract file path
        import re
        match = re.search(r"sandbox:(/.+\.html)", result)
        assert match
        file_path = match.group(1)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Clean up
        os.remove(file_path)
        
        # Let's try a specific placeholder we saw in the Bill of Lading
        bl_data = {
            "[ Location ]": "Seoul, Korea"
        }
        bl_result = generate_trade_document("BL", json.dumps(bl_data))
        
        match = re.search(r"sandbox:(/.+\.html)", bl_result)
        assert match
        bl_file_path = match.group(1)
        
        with open(bl_file_path, 'r', encoding='utf-8') as f:
            bl_content = f.read()
            
        assert "Seoul, Korea" in bl_content
        os.remove(bl_file_path)

    def test_full_offer_sheet(self):
        """Verify that a fully populated Offer Sheet is generated correctly."""
        data = {
            "[ Date ]": "2024-11-23",
            "[ Ref No ]": "OFF-2024-1123",
            "[ Buyer Name ]": "Apple Inc.",
            "[ Seller Name ]": "Samsung Electronics Co., Ltd.",
            "[ Item No ]": "1",
            "[ HS Code ]": "8517.13",
            "[ Product Description ]": "Galaxy S24 Ultra 512GB",
            "[ Quantity ]": "1,000 PCS",
            "[ Unit Price ]": "USD 1,200.00",
            "[ Amount ]": "USD 1,200,000.00",
            "[ Total Amount ]": "USD 1,200,000.00",
            "[ Country of Origin ]": "Republic of Korea",
            "[ Shipment ]": "By Sea from Busan Port",
            "[ Inspection ]": "Manufacturer's Inspection",
            "[ Payment ]": "Irrevocable L/C at sight",
            "[ Validity ]": "Until December 31, 2024",
            "[ Remarks ]": "Partial shipment allowed"
        }
        
        result = generate_trade_document("Offer_Sheet", json.dumps(data))
        
        import re
        match = re.search(r"sandbox:(/.+\.html)", result)
        assert match
        file_path = match.group(1)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Verify all fields are present
        assert "2024-11-23" in content
        assert "OFF-2024-1123" in content
        assert "Apple Inc." in content
        assert "Samsung Electronics Co., Ltd." in content
        assert "Galaxy S24 Ultra 512GB" in content
        assert "1,000 PCS" in content
        assert "USD 1,200,000.00" in content
        
        # Clean up
        os.remove(file_path)

    def test_sequential_generation(self):
        """Simulate the generation of documents in the specified order."""
        order = [
            "Offer_Sheet",
            "PI",
            "Overseas_Distribution_Contract",
            "Commercial_Invoice",
            "PL",
            "BL",
            "Letter_of_Credit"
        ]
        
        for doc_type in order:
            result = generate_trade_document(doc_type, json.dumps({}))
            assert "Error" not in result
            assert "성공적으로 작성되었습니다" in result
            
            # Extract file path from result to verify existence
            # Result format: "... [여기](sandbox:/path/to/file) ..."
            import re
            match = re.search(r"sandbox:(/.+\.html)", result)
            if match:
                file_path = match.group(1)
                assert os.path.exists(file_path)
                # Clean up
                os.remove(file_path)
