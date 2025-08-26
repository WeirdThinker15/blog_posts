from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os

def decrypt_file(encrypted_file_path, output_file_path, key):

    # Read IV + Encrypted Data
    with open(encrypted_file_path, 'rb') as fin:
        data = fin.read()

    iv = data[:16]  # First 16 bytes is IV
    encrypted_data = data[16:]

    # Create AES Cipher with the same Key and IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt Data
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove Padding
    padder = padding.PKCS7(128).padder()
    decrypted_data = padder.update(decrypted_padded) + padder.finalize()

    # Write the Original File
    with open(output_file_path, 'wb') as fout:
        fout.write(decrypted_data)

    print("Successfully decrypted file")

if __name__ == "__main__":

    # Read the Key from the file
    key_file_path = 'aes_key.txt'
    encrypted_file_path = 'sample.txt.enc'
    output_file_path = 'sample.txt.dec'

    # Read the Key from the file
    with open(key_file_path, 'rb') as fin:
        key = fin.read()

    decrypt_file(encrypted_file_path, output_file_path, key)
