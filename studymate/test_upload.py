#!/usr/bin/env python3
"""
Test script to verify the upload endpoint works
"""
import requests
import os

def test_upload():
    # Check if we have a test PDF
    test_files = [
        "data/uploads/aadhikar.pdf",
        "data/uploads/III aiml.pdf"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Testing upload with {test_file}")
            
            with open(test_file, 'rb') as f:
                files = {'file': f}
                try:
                    response = requests.post('http://localhost:8000/api/documents/upload', files=files)
                    print(f"Status Code: {response.status_code}")
                    print(f"Response: {response.text}")
                    return
                except Exception as e:
                    print(f"Error: {e}")
    
    print("No test files found")

if __name__ == "__main__":
    test_upload()
