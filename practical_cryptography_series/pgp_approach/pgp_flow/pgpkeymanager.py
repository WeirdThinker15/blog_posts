import gnupg
import os

class PGPKeyManager:

    def __init__(self, gpg_home="pgp_keys", logger=None):
        self.gpg_home = gpg_home
        self.logger = logger
        os.makedirs(gpg_home, exist_ok=True)
        self.gpg = gnupg.GPG(gnupghome=self.gpg_home,
                             use_agent=False,
                             options=["--pinentry-mode", "loopback"],
                             # gpgbinary="C:\\Program Files (x86)\\GnuPG\\bin\\gpg.exe",
                             verbose=False
                             )

        self.logger.debug(f" Public Keys : {self.gpg.list_keys()}")
        self.logger.debug(f" Private Keys : {self.gpg.list_keys(True)}")

    def generate_keys(self, name, email, passphrase):

        self.logger.info("Generating keys...")
        input_data = self.gpg.gen_key_input(
            name_real=name,
            name_email=email,
            key_type="RSA",
            key_length=2048,
            passphrase=passphrase,
        )

        key = self.gpg.gen_key(input_data)
        self.logger.info(f"Keys generated: {key}")
        return key.fingerprint

    def export_public_key(self, fingerprint, output_path):
        self.logger.info(f"Exporting public key : {output_path}")
        with open(output_path, "w") as f:
            f.write(
                self.gpg.export_keys(fingerprint)
            )

    def export_private_key(self, fingerprint, output_path, passphrase):
        self.logger.info(f"Exporting private key : {output_path}")
        with open(output_path, "w") as f:
            f.write(
                self.gpg.export_keys(fingerprint, passphrase=passphrase, secret=True)
            )

    def import_key(self, file_path):
        self.logger.info(f"Importing key : {file_path}")
        with open(file_path, "r") as f:
            return self.gpg.import_keys(f.read())