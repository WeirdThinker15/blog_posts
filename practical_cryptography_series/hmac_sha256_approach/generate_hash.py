import hmac
import hashlib

# Function to generate the HMAC SHA-256 Hash 
def generate_hash(file_path, secret_key):
    with open(file_path, 'rb') as fin:
        file_data = fin.read()
    
    hash = hmac.new(secret_key.encode(), file_data, hashlib.sha256).hexdigest()
    return hash

if __name__ == '__main__':
    file_path = 'sample.txt'
    
    # Read the Secret Key from the file 
    with open('secret_key.txt','r') as fin:
        secret_key = fin.read()

    # Generate the Hash of the File 
    file_hash = generate_hash(file_path=file_path, secret_key=secret_key)
    print(f"Hash of the File: {file_hash}")

    # Store the hash into a file 
    hash_file = f"{file_path}.hash"
    with open(hash_file, 'w') as fout:
        fout.write(file_hash)
    
    print("Hash Generated")
