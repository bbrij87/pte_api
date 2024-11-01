from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
from mod_logging import log_error  # Import error logging
import os
import base64
import json

# Load environment variables from .env file
load_dotenv()

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


def base64_urlsafe_encode(data):
    """Encode data in URL-safe base64."""
    return base64.urlsafe_b64encode(data).decode('utf-8')


def base64_urlsafe_decode(data):
    """Decode data from URL-safe base64."""
    return base64.urlsafe_b64decode(data)


def encrypt_data(data):
    """Encrypt data using AES-256-CBC with HMAC-SHA256 for integrity."""
    try:
        iv = os.urandom(16)
        padding_length = 16 - (len(data) % 16)
        padded_data = data + chr(padding_length) * padding_length

        cipher = Cipher(algorithms.AES(SECRET_KEY),
                        modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(
            padded_data.encode()) + encryptor.finalize()

        hmac_obj = hmac.HMAC(SECRET_KEY, hashes.SHA256(),
                             backend=default_backend())
        hmac_obj.update(iv + ciphertext)
        hmac_value = hmac_obj.finalize()

        encrypted_data = base64_urlsafe_encode(iv + ciphertext + hmac_value)
        return encrypted_data
    except Exception as e:
        log_error(f"Encryption failed: {str(e)}")
        raise ValueError("Encryption failed") from e


def decrypt_data(token):
    """Decrypt data encrypted with AES-256-CBC and HMAC-SHA256."""
    try:
        decoded_data = base64_urlsafe_decode(token)
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
    except Exception as e:
        log_error(f"Decryption failed: {str(e)}")
        raise ValueError("Decryption failed") from e


# Example usage
if __name__ == "__main__":
    sample_data = "Sensitive information"
    try:
        encrypted = encrypt_data(sample_data)
        print("Encrypted:", encrypted)

        decrypted = decrypt_data(encrypted)
        print("Decrypted:", decrypted)
    except Exception as e:
        log_error(f"Error in example usage: {str(e)}")
        print("An error occurred:", e)
