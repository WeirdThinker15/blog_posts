from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

# Generate RSA Public-Private Key
def generate_rsa_key(private_key_file, public_key_file):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Save Private Key
    with open(private_key_file, "wb") as fout:
        fout.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,             # Format = PEM
            format=serialization.PrivateFormat.TraditionalOpenSSL,  # Structure - OpenSSL style
            encryption_algorithm=serialization.NoEncryption()  # No password protection
        ))

    # Save Public Key
    public_key = private_key.public_key()
    with open(public_key_file, "wb") as fout:
        fout.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,        # Format = PEM
            format=serialization.PublicFormat.SubjectPublicKeyInfo      # Standard X.509 format
        ))

    print("RSA key generation complete")

# Generate ECDSA Key Pair
def generate_ec_key(private_key_file, public_key_file):

    # Generate ECDSA Private Key
    private_key = ec.generate_private_key(ec.SECP256R1())

    # Save Private Key
    with open(private_key_file, "wb") as fout:
        fout.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Save Public Key
    public_key = private_key.public_key()

    with open(public_key_file, "wb") as fout:
        fout.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    print("EC key generation complete")

if __name__ == "__main__":

    # Generating the RSA Key Pairs
    private_key_file_rsa = "rsa_private_key.pem"
    pubilic_key_file_rsa = "rsa_public_key.pem"
    # generate_rsa_key(private_key_file_rsa, pubilic_key_file_rsa)

    # Generating the ECDSA Key Pairs
    private_key_file_ec = "ecdsa_private_key.pem"
    public_key_file_ec = "ecdsa_public_key.pem"
    generate_ec_key(private_key_file_ec, public_key_file_ec)




