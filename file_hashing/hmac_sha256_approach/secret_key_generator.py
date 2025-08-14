import secrets
import base64

def generate_secret_key(key_outfile):
    # Generate a 32 byte(256-bit) random key
    key = secrets.token_bytes(32)
    encoded_key = base64.urlsafe_b64encode(key).decode()

    # Store the Key in a txt file for sharing it across 
    with open(key_outfile, "w") as fout:
        fout.write(encoded_key)
    
    print("Secret file generated")

if __name__ == "__main__":

    # Call function to generate the key 
    secret_file_name = "secret_key.txt"
    print("Generating the Secret Key")
    generate_secret_key(secret_file_name)
