param (
    [string]$File,
    [string]$HashFile
)

if (-not $File -or -not $HashFile) {
    Write-Output "Usage: .\verify_hash.ps1 -File <input_file> -HashFile <hash_file>"
    exit 1
}

if (-not (Test-Path $File)) {
    Write-Output "File '$File' not found."
    exit 1
}

if (-not (Test-Path $HashFile)) {
    Write-Output "Hash file '$HashFile' not found."
    exit 1
}

# Compute current hash
$computedHash = (Get-FileHash -Path $File -Algorithm SHA256).Hash.Trim()

# Read expected hash
$expectedHash = Get-Content $HashFile | Select-Object -First 1
$expectedHash = $expectedHash.Trim()

# Compare
if ($computedHash.ToUpper() -eq $expectedHash.ToUpper()) {
    Write-Output "Hash verified successfully."
} else {
    Write-Output "Hash mismatch!"
    Write-Output "Expected: $expectedHash"
    Write-Output "Actual:   $computedHash"
}
