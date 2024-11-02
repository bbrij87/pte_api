from utils.encryption import encrypt_data, decrypt_data
from utils.mod_logging import log_error, log_info

import sys
import os
import json

# command to run: sgb@SGBs-Laptop pte_api # python3 -m tests.test_encryption

# Import both functions to test encryption and decryption
# Sample data to test the API
test_data = {
    "request_id": "12345",  # Unique ID for this request
    "hostname": "example.com",  # Host making the request
    "username": "user123",  # Username of the requester
    "qid": "001",  # Question ID or related identifier
    "action": "template_checker",
    "qtype": "di",
    "qtext": "This is a sample text",  # Sample text for testing
    "image_type": "bar_chart",  # Type of image being referenced
    "response_text": "The bar graph image is about vehicle colour involved in total-loss collision...",
    "tags": "vehicle, collision, frequency, accident, blue, green, red, white, black, grey, high, low, comparison, involvement, statistics, traffic, safety, analysis"
}


# Convert data to JSON format
json_data = json.dumps(test_data)

# Encrypt the JSON data
encrypted_data = encrypt_data(json_data)
encrypted_data2 = '''
-D7xcTQVzJKB3N2tWBoWsQ0ff7xYwmHy_7mLKARlysKSINwhM050kWkzhR-Yx-MXaV9i0tmkJVzVo8FXkzJvXf9Gj4XyzfjaKuHpFnB7PLPwx__QXDls4uzBSZTDwa9nfDFlIszIybpxMepWPRSAyi6czm4Q_G7rBSOjthNAHjBxYiDcw5jV-IajHeLhJ97qkI4zJY7w_Fy_V2A-dnHjSQPTgzh6ouSsg5jJfKD1AY9vmlMcF_moTZ6FcPyzG4Wo80NMRp_7eA6rDh_dyluNOJoIZsTVCNqzLwJrbRmd-Hh2Fy35AGVyQjThluMOX2rXqOgrHfVG9SzLrHn6-LLoY7kFYfyFHSoEVaOgwv2Fi7NMW-IiqDgdRfSPjZPoUazYCDGuSzTSyiVX19vEzT3pWBFCfbiOo-aWm8l7euIbiPRdcSu1z3eLXgVX7PTc9mYwwiRW85w5PJlEcNEAuMuJyeTvgnHB79qnyLyMwdgqHmVvt-5iWnSUSpQQp_BzXAhAprq7OiGzSzqaf7KVAzf-0i0_6EjnXj7ZoTZEOGFyTugx6t4QbZYqMYpjDmnsWU-N9FVamMyHZ-QeYWT1rZZ8eNciB1YiFEsN1W9deZyqcQDFBcB_mjdJPKNtmkeal-D-GNbDDx6JmUY9iQSrG-_oh-emWXDSHk7Hg6YQ5YxMpJA=
'''
encrypted_data3 = '''
krDUsScPY5_0mwNC1mpUTDmx7cxss60kgN7pqr0N_N--UYOY6pOYx5YrvO3OJFfjxy08GLz11Emm6QDiSmMgLFguALOn2s5OSVch7XWLx34DiNKmfvS7_1REXiZxdvtz
'''


print("Encrypted Data:", encrypted_data3)

# Decrypt to verify
try:
    decrypted_data = decrypt_data(encrypted_data3)
    # Convert decrypted JSON string back to dictionary
    decrypted_json = json.loads(decrypted_data)
    print("Decrypted Data:", decrypted_json)
except Exception as e:
    print("Decryption failed:", e)
