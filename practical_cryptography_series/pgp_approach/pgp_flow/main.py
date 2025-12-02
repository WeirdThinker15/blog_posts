from pgpdecryption import PGPDecryptor
from pgpencryptor import PGPEncryptor
from pgpkeymanager import PGPKeyManager
from setup_logging import setup_logging

if __name__ == "__main__":

    LOGGER = setup_logging()

    operation = input("Enter the operation to perform: 1 for key generation, 2 for encryption, 3 for decryption: ")
    manager = PGPKeyManager(gpg_home="resources/pgp_keys3",logger=LOGGER)
    operation = int(operation)
    if operation == 1:

        # Generate Key Pairs
        # For Sender
        sender_fp = manager.generate_keys("Sender", "sender@bank.com", "sender123")

        # For Receiver
        receiver_fp = manager.generate_keys("Receiver", "receiver@bank.com", "receiver123")

        # Export the Public and Private Keys
        # For Sender
        manager.export_public_key(sender_fp, output_path="resources/sender_public.asc")
        manager.export_private_key(sender_fp, output_path="resources/sender_private.asc",passphrase="sender123")

        # For Receiver
        manager.export_public_key(receiver_fp, output_path="resources/receiver_public.asc")
        manager.export_private_key(receiver_fp, output_path="resources/receiver_private.asc", passphrase="receiver123")

    elif operation == 2:
        # Let try encrypting and decrypting a sample file
        # Sample File
        with open("bank_data.txt", "w") as f:
            f.write("Sample file for PGP Encryption")

        # Encrypt the File
        encryptor = PGPEncryptor(manager.gpg,
                                 receiver_public_key="resources/receiver_public.asc",
                                 sender_private_key="resources/sender_private.asc",
                                 sender_passphrase="sender123",
                                 logger=LOGGER
                                 )
        result = encryptor.encrypt_file(
            input_file_path="bank_data.txt",
            output_file_path="bank_data.txt.pgp"
        )

    elif operation == 3:
        # Lets Decrypt and Verify Signature of the Encrypted File
        decryptor = PGPDecryptor(manager.gpg,
                                 receiver_private_key="resources/receiver_private.asc",
                                 sender_public_key="resources/sender_public.asc",
                                 receiver_passphrase="receiver123",
                                 logger=LOGGER
                                 )

        # Perform the Operation
        decryptor.decrypt_file(
            input_file_path="bank_data.txt.pgp",
            output_file_path="decrypted_bank_data.txt.",
        )




