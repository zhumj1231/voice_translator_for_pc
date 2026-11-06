$ErrorActionPreference = 'Stop'
$taskSecret = Read-Host 'Paste your xAI API key (hidden input)' -AsSecureString
$taskPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecret)
try {
    $taskApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer).Trim()
    if ($taskApiKey.Length -lt 16 -or $taskApiKey -match '\s') {
        throw 'Invalid key: paste the complete key with no spaces.'
    }
    [Environment]::SetEnvironmentVariable('XAI_API_KEY', $taskApiKey, 'User')
    Write-Host 'Saved XAI_API_KEY in your Windows user environment. No project file contains the key.'
    Write-Host 'Close and reopen the subtitle tool after setup.'
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)
    $taskApiKey = $null
    $taskSecret.Dispose()
}
Read-Host 'Press Enter to close'
