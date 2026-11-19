[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$CheckOnly,
    [switch]$OfflineOnly
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$compose = @('compose', '-p', 'researchflow', '-f', (Join-Path $root 'docker-compose.full.yml'),
    '-f', (Join-Path $root 'docker-compose.evaluation.override.yml'))
if (-not $OfflineOnly) { $compose += @('--profile', 'eval-online') }

function Invoke-DockerChecked {
    param([string[]]$DockerArguments)
    # Windows PowerShell can wrap progress on stderr as an error; use the process code.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & docker @DockerArguments
        if ($LASTEXITCODE -ne 0) { throw "Docker exited with code $LASTEXITCODE" }
    }
    finally { $ErrorActionPreference = $previousPreference }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker is not installed.' }
if (-not (Test-Path -LiteralPath (Join-Path $root '.env') -PathType Leaf)) { throw 'Missing .env; no file was created or replaced.' }

Push-Location -LiteralPath $root
try {
    Invoke-DockerChecked -DockerArguments @('version', '--format', '{{.Server.Version}}')
    $raw = & docker @compose config --format json
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the full-stack evaluation configuration.' }
    $config = ($raw -join "`n") | ConvertFrom-Json
    if ($config.services.postgres.image -ne 'pgvector/pgvector:pg16' -or
        $config.volumes.postgres_data.name -ne 'mara_full_postgres_data' -or
        $config.volumes.redis_data.name -ne 'mara_full_redis_data' -or
        $config.volumes.corpus_data.name -ne 'mara_full_corpus_data' -or
        $config.volumes.rf_eval_job_data.name -ne 'researchflow_eval_job_data') {
        throw 'Unexpected database image or volume identity; refusing to start.'
    }
    if ($config.services.postgres.ports -or $config.services.redis.ports) {
        throw 'Full-stack datastores must not publish host ports.'
    }
    Write-Host 'Full-stack evaluation configuration verified; original named volumes retained.'
    if (-not $CheckOnly) {
        if ($Build) { Invoke-DockerChecked -DockerArguments ($compose + @('build')) }
        Invoke-DockerChecked -DockerArguments ($compose + @('up', '-d', '--no-build', '--wait', '--wait-timeout', '180'))
        $apps = @('api', 'worker', 'frontend', 'eval-pipeline-offline')
        if (-not $OfflineOnly) { $apps += 'eval-pipeline-online' }
        $stale = @()
        foreach ($service in $apps) {
            $running = & docker inspect "researchflow-$service-1" --format '{{.Image}}'
            if ($LASTEXITCODE -ne 0) { throw "Cannot inspect running service $service" }
            $built = & docker image inspect "researchflow-$service" --format '{{.Id}}'
            if ($LASTEXITCODE -ne 0) { throw "Cannot inspect built image for $service" }
            if ($running.Trim() -ne $built.Trim()) { $stale += $service }
        }
        # A matching Compose configuration does not prove it uses the newly tagged image.
        # Align application images explicitly; never force-recreate the data services here.
        if ($stale.Count -gt 0) {
            Invoke-DockerChecked -DockerArguments ($compose + @('up', '-d', '--no-build', '--no-deps', '--force-recreate', '--wait', '--wait-timeout', '180') + $stale)
        }
        Invoke-DockerChecked -DockerArguments ($compose + @('ps'))
        Write-Host 'ResearchFlow is ready. Open http://localhost:3031 (or your configured FRONTEND_PORT).'
    }
}
finally { Pop-Location }
