import hmac 
from generate_hash import generate_hash

# Function to verify the hash of the given file 
def verify_hash(file_path, secret_key, file_hash):

    # Compute the Hash of the File 
    calculated_hash = generate_hash(file_path=file_path, secret_key=secret_key)

    # Match the hash with the received hash 
    is_hash_valid = hmac.compare_digest(calculated_hash, file_hash)
    return is_hash_valid

if __name__ == "__main__":

    file_path = 'sample.txt'
    file_hash_path = 'sample.txt.hash'
    with open(file_hash_path, 'r') as fin:
        file_hash = fin.read()
    
    with open('secret_key.txt', 'r') as fin:
        secret_key = fin.read()
    
    # Verify the hash 
    is_hash_valid = verify_hash(
        file_path=file_path,
        secret_key=secret_key,
        file_hash=file_hash
    )

    print(f"Hash Verification Result : {is_hash_valid}")
