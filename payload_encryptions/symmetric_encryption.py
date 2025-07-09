from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import json
import base64

# Step 1 : Generate a 256-bit (32 byte) symmetric key
def generate_symmetric_key():
    return AESGCM.generate_key(bit_length=256)

# Step 2 : Encrypt the JSON payload using the symmetric key
def encrypt_json_payload(plaintext, symmetric_key):
    aesgcm = AESGCM(symmetric_key)
    nounce = os.urandom(12)  # 96 bit nouce required for AES-GCM
    json_payload = json.dumps(plaintext).encode('utf-8')
    ciphertext = aesgcm.encrypt(nounce, json_payload, None)
    return {
        'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
        'nonce'     : base64.b64encode(nounce).decode('utf-8'),
        }

# Step 3 : Decrypt the JSON payload
def decrypt_json_payload(encrypted_text, symmetric_key):
    aesgcm = AESGCM(symmetric_key)
    nounce = base64.b64decode(encrypted_text['nonce'])
    ciphertext = base64.b64decode(encrypted_text['ciphertext'])
    plaintext = aesgcm.decrypt(nounce, ciphertext, None)
    return json.loads(plaintext.decode('utf-8'))


if __name__ == "__main__":
    payload = {
        "name": "xyz",
        "amount": 1000
    }

    key = generate_symmetric_key() 
    print(f"Generated Symmetric Key : {key}")

    print(" Iteration 1 :")
    encrypted_text = encrypt_json_payload(payload, key)
    print(f"Encrypted Payload : {json.dumps(encrypted_text, indent=2)}")

    decrypted_text = decrypt_json_payload(encrypted_text, key)
    print(f"Decrypted Payload : {json.dumps(decrypted_text, indent=2)}")

    print("Iteration 2 ")
    encrypted_text = encrypt_json_payload(payload, key)
    print(f"Encrypted Payload : {json.dumps(encrypted_text, indent=2)}")

    decrypted_text = decrypt_json_payload(encrypted_text, key)
    print(f"Decrypted Payload : {json.dumps(decrypted_text, indent=2)}")
