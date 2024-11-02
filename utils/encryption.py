from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
from utils.mod_logging import log_error
import os
import base64
import json

# command to run: sgb@SGBs-Laptop pte_api % python3 -m utils.encryption
# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join('secret', '.env'))

# Fetch the Base64-encoded secret key from environment variables
encoded_key = os.getenv("SECRET_KEY")

# Decode the key to get a 32-byte key for AES-256
if encoded_key:
    try:
        SECRET_KEY = base64.urlsafe_b64decode(encoded_key)
        if len(SECRET_KEY) != 32:
            raise ValueError("SECRET_KEY is not 32 bytes.")
    except Exception as e:
        log_error(f"Invalid SECRET_KEY: {str(e)}")
        raise ValueError("Invalid SECRET_KEY") from e
else:
    log_error("SECRET_KEY not found in environment variables.")
    raise ValueError("SECRET_KEY not found")


def base64_encode_with_padding(data):
    """Encode data in base64 with enforced padding."""
    encoded_data = base64.urlsafe_b64encode(data)
    # Ensure the encoded data length is a multiple of 4 by adding `=` padding
    return encoded_data + b'=' * (-len(encoded_data) % 4)


def base64_decode_with_padding(data):
    """Decode base64 data with ensured padding."""
    # Add `=` padding if necessary
    padded_data = data + b'=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded_data)


def encrypt_data(data):
    """Encrypt data using AES-256-CBC with HMAC-SHA256 for integrity."""
    iv = os.urandom(16)
    padding_length = 16 - (len(data) % 16)
    padded_data = data + chr(padding_length) * padding_length

    cipher = Cipher(algorithms.AES(SECRET_KEY),
                    modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data.encode()) + encryptor.finalize()

    hmac_obj = hmac.HMAC(SECRET_KEY, hashes.SHA256(),
                         backend=default_backend())
    hmac_obj.update(iv + ciphertext)
    hmac_value = hmac_obj.finalize()

    # Use the new base64 encoding function to enforce padding
    encrypted_data = base64_encode_with_padding(iv + ciphertext + hmac_value)
    # Convert bytes to string for easier handling
    return encrypted_data.decode('utf-8')


def decrypt_data(token):
    """Decrypt data encrypted with AES-256-CBC and HMAC-SHA256."""
    # Decode with ensured padding
    decoded_data = base64_decode_with_padding(token.encode('utf-8'))

    iv = decoded_data[:16]
    ciphertext = decoded_data[16:-32]
    hmac_value = decoded_data[-32:]

    hmac_obj = hmac.HMAC(SECRET_KEY, hashes.SHA256(),
                         backend=default_backend())
    hmac_obj.update(iv + ciphertext)
    hmac_obj.verify(hmac_value)

    cipher = Cipher(algorithms.AES(SECRET_KEY),
                    modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    padding_length = padded_data[-1]
    data = padded_data[:-padding_length].decode()
    return data


# Example usage
if __name__ == "__main__":
    # Test data
    test_data = json.dumps({"sample": "data"})
    encrypted = encrypt_data(test_data)
    print("Encrypted:", encrypted)

    decrypted = decrypt_data(encrypted)
    print("Decrypted:", decrypted)
