from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os

def encrypt_file(file_path, encrypted_file_path, key, iv):

    # Read File Data
    with open(file_path, 'rb') as fin:
        data = fin.read()

    # Pad the Data (AES needs data in blocks)
    padder = padding.PKCS7(128).padder()   # 128-bit AES Block Size
    padded_data = padder.update(data) + padder.finalize()

    # Create AES Cipher in CBC Mode
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Encrypt the data
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Write IV + Encrypted Data to the Output File
    with open(encrypted_file_path, 'wb') as fout:
        fout.write(iv + encrypted_data)

    print("Encrypted file saved to {}".format(encrypted_file_path))

if __name__ == '__main__':

    # Generate a random AES key (256-bit for strong encryption)
    key = os.urandom(32)  # 32 bytes = 256-bits

    # Write the Key to a file
    with open("aes_key.txt", "wb") as fout:
        fout.write(key)

    # Generate a random Initialization Vector (IV) for CBC Mode
    iv = os.urandom(16)   # 16 bytes for AES Block Size

    # Provide the Input File and Output File Paths
    file_path = 'sample.txt'
    output_file_path = 'sample.txt.enc'

    encrypt_file(file_path, output_file_path, key, iv)

