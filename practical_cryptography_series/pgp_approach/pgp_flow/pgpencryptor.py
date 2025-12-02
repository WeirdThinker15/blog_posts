import os
import re

class PGPEncryptor:

    def __init__(self, gpg,receiver_public_key, sender_private_key, sender_passphrase, logger):
        self.gpg = gpg
        self.logger = logger
        self.sender_passphrase = sender_passphrase

        with open(receiver_public_key, "rb") as key_file:
            receiver_key_data = key_file.read()

        with open(sender_private_key, "rb") as key_file:
            sender_key_data = key_file.read()

        self.receiver_key = self._import_key(receiver_key_data)
        self.sender_key = self._import_key(sender_key_data)
        self.logger.debug(f"Receiver public key: {self.receiver_key}")
        self.logger.debug(f"Sender public key: {self.sender_key}")

    def _import_key(self, key_data):
        self.logger.info(f"Importing key...")
        imported = self.gpg.import_keys(key_data)
        self.logger.info(f"Imported : {imported.__dict__}")
        if not imported.fingerprints:
            self.logger.error(f"key import failed!")
            raise Exception(f"key import failed!")

        self.logger.info(f"key imported successfully")
        return imported.fingerprints[0]

    def encrypt_file(self, input_file_path, output_file_path):

        self.logger.info(f"Starting encryption for file: {input_file_path}")

        with open(input_file_path, "rb") as input_file:

            status = self.gpg.encrypt_file(
                input_file,
                recipients=[self.receiver_key],
                sign=True,
                passphrase=self.sender_passphrase,
                always_trust=True,
                output=output_file_path,
                # extra_args=["--pinentry-mode", "loopback"]
                extra_args = ["--pinentry-mode", "loopback", "--local-user", self.sender_key]
            )

        is_sig_created = False
        if status.ok:
            self.logger.info(f"Encryption Successful : {status.ok}")
            # Custom code to check if signature is generated or not
            # Since the current library does not parse the status correctly
            for line in status.stderr.splitlines():
                if "SIG_CREATED" in line:
                    self.logger.info(f"SIG_CREATED: {line}")
                    is_sig_created = True
                    break

            if not is_sig_created:
                self.logger.error(f"Signature not created")

        else:
            self.logger.error(f"Encryption Failure : {status.stderr}")
        return {
            "ok": status.ok,
            "status": status.status,
            "stderr": status.stderr,
        }