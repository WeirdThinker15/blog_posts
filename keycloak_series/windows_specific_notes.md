## Recommended Folder Structure

```
project/
├── docker-compose.yml
├── ca/
│   ├── ca.key
│   └── ca.crt
├── server/
│   ├── server.key
│   ├── server.csr
│   ├── server.crt
│   ├── server.ext
│   └── truststore.p12
├── client/
│   ├── client.key
│   ├── client.csr
│   ├── client.crt
│   ├── client.ext
│   └── client.p12
```

## Step 1 — Generate CA Certificate

### Generate CA Key 
```
openssl genrsa -out ca/ca.key 2048
```

### Generate CA Certificate
```
openssl req -x509 -new -nodes \
  -key ca/ca.key \
  -sha256 \
  -days 365 \
  -out ca/ca.crt \
  -subj "//CN=Local-CA"
```

## Step 2 — Create Server Extension File

```
touch server/server.ext
```

### File Contents
```
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:localhost
```

*This is required because modern TLS clients validate SAN instead of CN.*

## Step 3 — Generate Server Certificate

### Generate Server Key 
```
openssl genrsa -out server/server.key 2048
```

### Generate Server CSR
```
openssl req -new \
  -key server/server.key \
  -out server/server.csr \
  -subj "//CN=localhost"
```

### Sign Server Certificate
```
openssl x509 -req \
  -in server/server.csr \
  -CA ca/ca.crt \
  -CAkey ca/ca.key \
  -CAcreateserial \
  -out server/server.crt \
  -days 365 \
  -sha256 \
  -extfile server/server.ext
```

### Verify Server Certificate
```
openssl x509 -in server/server.crt -text -noout
```
## Step 4 — Create Client Extension File

```
touch client/client.ext
```

### File Contents
```
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=clientAuth
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
```

*This enables the certificate for TLS client authentication.*

## Step 5 — Generate Client Certificate

### Generate Client Key
```
openssl genrsa -out client/client.key 2048
```

### Generate Client CSR
*Important : Client CN should match Keycloak Client ID.*

```
# If Client ID=arblogs

openssl req -new \
  -key client/client.key \
  -out client/client.csr \
  -subj "//CN=arblogs"
```

### Sign Client Certificate
```
openssl x509 -req \
  -in client/client.csr \
  -CA ca/ca.crt \
  -CAkey ca/ca.key \
  -CAcreateserial \
  -out client/client.crt \
  -days 365 \
  -sha256 \
  -extfile client/client.ext
```

### Verify Client Certificate
```
openssl x509 -in client/client.crt -text -noout
```

## Step 6 — Export Client PKCS12 Certificate

*Windows curl/Postman works better with PKCS12.*

### Generate P12
```
openssl pkcs12 -export \
  -legacy \
  -out client/client.p12 \
  -inkey client/client.key \
  -in client/client.crt \
  -certfile ca/ca.crt
```

## Step 7 — Create Keycloak Truststore
*Keycloak must trust the CA certificate*

### Generate Truststore

```
keytool -importcert \
  -file ca/ca.crt \
  -alias local-ca \
  -keystore server/truststore.p12 \
  -storetype PKCS12 \
  -storepass password \
  -noprompt
```

## Step 8 — Docker Compose
*Create docker-compose.yml File*

```
version: '3.8'

services:
  keycloak:
    image: quay.io/keycloak/keycloak:26.2.5
    container_name: keycloak-mtls

    command:
      - start-dev
      - --https-certificate-file=/opt/keycloak/certs/server/server.crt
      - --https-certificate-key-file=/opt/keycloak/certs/server/server.key
      - --https-client-auth=request
      - --https-trust-store-file=/opt/keycloak/certs/server/truststore.p12
      - --https-trust-store-password=password
      - --https-protocols=TLSv1.2

    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin

      KC_HOSTNAME: localhost
      KC_HOSTNAME_STRICT: true
      KC_HTTP_ENABLED: false

    ports:
      - "8443:8443"

    volumes:
      - ./ca:/opt/keycloak/certs/ca
      - ./server:/opt/keycloak/certs/server
      - ./client:/opt/keycloak/certs/client
```

## Step 9 — Start Keycloak
```
docker-compose up -d
```

## Step 10 — Verify HTTPS
```
curl.exe -v --tlsv1.2 --ssl-no-revoke `
  --cacert ca/ca.crt `
  https://localhost:8443
```

## Step 11 — Configure Keycloak Client

Inside KeyCloak
- Create Client with Client ID = arblogs
- Under Settings : Enable 
    - Client Authentication
    - Service Account Enabled
- Set Credentials Client Authenticator = X509 Certificate
- X509 Settings :
    - Allow Regex Comparison 
    - Regex : CN=(.*?)(?:,|$)

## Step 12 — Configure Postman

1. URL : https://localhost:8443/realms/master/protocol/openid-connect/token
2. Body 

| Key        | Value              |
| ---------- | ------------------ |
| grant_type | client_credentials |
| client_id  | arblogs            |

3. Client Certificate : Under Settings -> Certificates 

| Field      | Value             |
| ---------- | ----------------- |
| Host       | localhost         |
| Port       | 8443              |
| PFX File   | client/client.p12 |
| Passphrase | keycloak          |



