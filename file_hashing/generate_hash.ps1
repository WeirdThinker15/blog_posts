param (
    [Parameter(Mandatory = $true)]
    [string]$File,

    [string]$HashFile
)

# Check if the input file exists
if (-not (Test-Path $File)) {
    Write-Error "Input file '$File' not found."
    exit 1
}

# Set default hash file if not provided
if (-not $HashFile) {
    $HashFile = "$File.sha256"
}

try {
    # Compute SHA256 hash
    $hash = Get-FileHash -Path $File -Algorithm SHA256
    $hash.Hash | Out-File -FilePath $HashFile -Encoding ascii -Force
    Write-Output "SHA256 hash written to '$HashFile'"
    exit 0
}
catch {
    Write-Error "Failed to generate hash: $_"
    exit 2
}
