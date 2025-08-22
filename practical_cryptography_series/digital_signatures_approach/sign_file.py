from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def generate_digital_signature(private_key_file, file_path, signature_file_path):

    # Load File Content
    with open(file_path, "rb") as fin:
        data = fin.read()

    # Read the Private Key from pem file
    with open(private_key_file, "rb") as fout:
        private_key = serialization.load_pem_private_key(
            fout.read(),
            password=None
        )

    # Sign the Data
    signature = private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    # Save the Signature
    with open(signature_file_path, "wb") as fout:
        fout.write(signature)

    print("Signature generation complete")

if __name__ == "__main__":

    # Generate Digital Signature
    private_key_file = "rsa_private_key.pem"
    file_path = "sample_2.txt"
    signature_file_path = "sample_2.signature"
    generate_digital_signature(private_key_file, file_path, signature_file_path)
