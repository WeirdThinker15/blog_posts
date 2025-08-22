from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# Verify the File with the Signature
def verify_file(public_key_file, file_path, file_signature_path):

    # Load Public Key
    with open(public_key_file, "rb") as fin:
        public_key = serialization.load_pem_public_key(
            fin.read(),
            backend=default_backend()
        )

    # Load File Signature
    with open(file_signature_path, "rb") as fin:
        signature = fin.read()

    # Load File Content
    with open(file_path, "rb") as fin:
        data = fin.read()

    # Verify the Signature
    try:
        public_key.verify(
            signature=signature,
            data=data,
            padding=padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            algorithm=hashes.SHA256()
        )

        print("Signature verified")
    except Exception as e:
        print("Signature verification failed")
        print(f"Exception: {e}")


if __name__ == '__main__':

    # Verify the File Signature
    public_key_file = "rsa_public_key.pem"
    file_path = 'sample.txt'
    file_signature_path = 'sample_2.signature'
    verify_file(
        public_key_file=public_key_file,
        file_path=file_path,
        file_signature_path=file_signature_path
    )
