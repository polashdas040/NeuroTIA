param([int]$Port = 4174)

$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCommand) {
    $uiNodeExecutable = $nodeCommand.Source
} else {
    $uiNodeExecutable = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
}
if (-not (Test-Path -LiteralPath $uiNodeExecutable)) {
    throw 'Node.js was not found. Install Node.js 18 or later, then run node web_ui/server.js.'
}
$env:PORT = "$Port"
& $uiNodeExecutable (Join-Path $PSScriptRoot 'server.js')
