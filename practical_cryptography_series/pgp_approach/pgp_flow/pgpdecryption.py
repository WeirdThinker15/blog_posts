
class PGPDecryptor:

    def __init__(self, gpg,receiver_private_key, sender_public_key, receiver_passphrase,logger):
        self.gpg = gpg
        self.logger = logger
        self.receiver_private_key = receiver_private_key
        self.sender_public_key = sender_public_key
        self.receiver_passphrase = receiver_passphrase

        # Import the Keys
        self.receiver_fingerprint = self._import_key(receiver_private_key)
        self.sender_fingerprint = self._import_key(sender_public_key)

    def _import_key(self, key_path):

        self.logger.info(f"Importing key from {key_path}")

        with open(key_path, "rb") as key_file:
            key_data = key_file.read()

        imported = self.gpg.import_keys(key_data)
        if not imported.fingerprints:
            self.logger.error("Key import failed!")
            raise Exception("Key import failed")

        self.logger.info("Key imported successfully")
        return imported.fingerprints[0]

    def decrypt_file(self, input_file_path,output_file_path):

        self.logger.info(f"Starting decryption for file: {input_file_path}")

        with open(input_file_path, "rb") as input_file:
            result = self.gpg.decrypt_file(
                input_file,
                passphrase=self.receiver_passphrase,
                output=output_file_path,
            )

        if not result.ok:
            self.logger.error(f"Failed to decrypt: {input_file_path}, Error : {result.stderr}")
            raise Exception("Failed to decrypt")
        else:
            self.logger.info(f"Decryption Successful {result.ok}")
            self.logger.info(f"Signature Valid : {result.valid}")
            self.logger.info(f"Signed By: {result.username}")

        self.logger.info(f"Finished decryption for file: {input_file_path}")



