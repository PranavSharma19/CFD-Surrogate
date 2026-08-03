param(
    [string]$Destination = "C:\Repositories\steve\data\raw\stanford_vmr",
    [int]$ParallelDownloads = 4,
    [double]$SafetyReserveGB = 25
)

$ErrorActionPreference = "Stop"

$manifestDir = Join-Path $Destination "manifests"
$projectDir = Join-Path $Destination "projects"
$surfaceDir = Join-Path $Destination "surface_results"

New-Item -ItemType Directory -Force -Path $manifestDir, $projectDir, $surfaceDir | Out-Null

$sizesUrl = "https://www.vascularmodel.com/dataset/file_sizes.csv"
$projectsUrl = "https://www.vascularmodel.com/dataset/dataset-svprojects.csv"
$resultsUrl = "https://www.vascularmodel.com/dataset/dataset-svresults.csv"

$sizesPath = Join-Path $manifestDir "file_sizes.csv"
$projectsPath = Join-Path $manifestDir "dataset-svprojects.csv"
$resultsPath = Join-Path $manifestDir "dataset-svresults.csv"

curl.exe -L --fail --retry 5 --silent --show-error --output $sizesPath $sizesUrl
curl.exe -L --fail --retry 5 --silent --show-error --output $projectsPath $projectsUrl
curl.exe -L --fail --retry 5 --silent --show-error --output $resultsPath $resultsUrl

$sizes = Import-Csv $sizesPath
$projects = Import-Csv $projectsPath
$results = Import-Csv $resultsPath
$caseIds = 31..45 | ForEach-Object { "{0:D4}_H_ABAO_AAA" -f $_ }

$eligibleProjects = @($projects | Where-Object {
    $caseIds -contains $_.Name -and $_.Results -eq "1" -and $_.Procedure -eq "None"
})

if ($eligibleProjects.Count -ne 15) {
    throw "Expected 15 eligible Stanford AAA projects, found $($eligibleProjects.Count)."
}

$eligibleResults = @($results | Where-Object {
    $caseIds -contains $_."Model Name" -and
    $_."Results File Type" -eq "Surface (vtp)" -and
    $_."Results Type" -eq "Time-Resolved" -and
    $_."Simulation Method" -eq "Rigid Wall"
})

if ($eligibleResults.Count -ne 15) {
    throw "Expected 15 eligible Stanford surface-result archives, found $($eligibleResults.Count)."
}

$downloads = @()
foreach ($project in $eligibleProjects) {
    $relativePath = "svprojects/$($project.Name).zip"
    $sizeRow = $sizes | Where-Object { $_.Name -eq $relativePath }
    if (-not $sizeRow) { throw "Missing size metadata for $relativePath" }
    $downloads += [pscustomobject]@{
        CaseId = $project.Name
        Kind = "project"
        Url = "https://www.vascularmodel.com/$relativePath"
        Destination = Join-Path $projectDir "$($project.Name).zip"
        ExpectedBytes = [int64]$sizeRow.Size
    }
}

foreach ($result in $eligibleResults) {
    $relativePath = "svresults/$($result.'Model Name')/$($result.'Full Simulation File Name')"
    $sizeRow = $sizes | Where-Object { $_.Name -eq $relativePath }
    if (-not $sizeRow) { throw "Missing size metadata for $relativePath" }
    $downloads += [pscustomobject]@{
        CaseId = $result."Model Name"
        Kind = "surface_result"
        Url = "https://www.vascularmodel.com/$relativePath"
        Destination = Join-Path $surfaceDir $result."Full Simulation File Name"
        ExpectedBytes = [int64]$sizeRow.Size
    }
}

# The repository's file_sizes.csv can lag behind replaced archives. Resolve
# the live Content-Length for every target before storage checks or validation.
foreach ($download in $downloads) {
    $headers = & curl.exe -L --fail --silent --show-error --head --max-time 60 $download.Url
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read HTTP headers for $($download.Url)"
    }
    $contentLengthLines = @($headers | Where-Object { $_ -match '^Content-Length:\s*(\d+)\s*$' })
    if ($contentLengthLines.Count -lt 1) {
        throw "No Content-Length returned for $($download.Url)"
    }
    $contentLengthLines[-1] -match '^Content-Length:\s*(\d+)\s*$' | Out-Null
    $download.ExpectedBytes = [int64]$Matches[1]
}

# Promote previously completed partial files using the live server size.
foreach ($download in $downloads) {
    $partialPath = "$($download.Destination).part"
    if (Test-Path -LiteralPath $partialPath) {
        $partialBytes = (Get-Item -LiteralPath $partialPath).Length
        if ($partialBytes -gt $download.ExpectedBytes) {
            throw "Partial file exceeds live server size: $partialPath"
        }
        if ($partialBytes -eq $download.ExpectedBytes) {
            Move-Item -LiteralPath $partialPath -Destination $download.Destination
        }
    }
}

$remainingBytes = 0L
foreach ($download in $downloads) {
    if (Test-Path -LiteralPath $download.Destination) {
        $existingSize = (Get-Item -LiteralPath $download.Destination).Length
        if ($existingSize -eq $download.ExpectedBytes) { continue }
        throw "Existing file has the wrong size: $($download.Destination)"
    }

    $partialPath = "$($download.Destination).part"
    $partialSize = if (Test-Path -LiteralPath $partialPath) {
        (Get-Item -LiteralPath $partialPath).Length
    } else { 0L }
    $remainingBytes += [math]::Max(0L, $download.ExpectedBytes - $partialSize)
}

$driveName = (Split-Path -Qualifier $Destination).TrimEnd(":")
$freeBytes = (Get-PSDrive -Name $driveName).Free
$reserveBytes = [int64]($SafetyReserveGB * 1GB)
if (($freeBytes - $remainingBytes) -lt $reserveBytes) {
    throw "Insufficient storage: download needs $([math]::Round($remainingBytes/1GB, 2)) GB and must preserve a $SafetyReserveGB GB reserve."
}

$inventoryPath = Join-Path $manifestDir "download_inventory.json"
$downloads | Select-Object CaseId, Kind, Url, Destination, ExpectedBytes |
    ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $inventoryPath

$pending = @($downloads | Where-Object {
    -not (Test-Path -LiteralPath $_.Destination) -or
    (Get-Item -LiteralPath $_.Destination).Length -ne $_.ExpectedBytes
})

if ($pending.Count -gt 0) {
    $curlArgs = @(
        "--parallel", "--parallel-max", "$ParallelDownloads",
        "-L", "--fail", "--retry", "5", "--retry-delay", "3",
        "--continue-at", "-", "--show-error"
    )
    foreach ($download in $pending) {
        $curlArgs += @(
            "--url", $download.Url,
            "--output", "$($download.Destination).part"
        )
    }

    & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "One or more Stanford downloads failed. Re-run this script to resume."
    }
}

foreach ($download in $downloads) {
    $partialPath = "$($download.Destination).part"
    if (Test-Path -LiteralPath $partialPath) {
        $actualBytes = (Get-Item -LiteralPath $partialPath).Length
        if ($actualBytes -ne $download.ExpectedBytes) {
            throw "Size verification failed for ${partialPath}: expected $($download.ExpectedBytes), got $actualBytes."
        }
        Move-Item -LiteralPath $partialPath -Destination $download.Destination
    }

    if (-not (Test-Path -LiteralPath $download.Destination)) {
        throw "Missing completed download: $($download.Destination)"
    }
    $actualBytes = (Get-Item -LiteralPath $download.Destination).Length
    if ($actualBytes -ne $download.ExpectedBytes) {
        throw "Size verification failed for $($download.Destination)."
    }
}

$summary = [pscustomobject]@{
    CompletedAt = (Get-Date).ToString("o")
    Source = "Stanford Vascular Model Repository"
    CaseCount = $eligibleProjects.Count
    ProjectArchiveCount = $eligibleProjects.Count
    SurfaceResultArchiveCount = $eligibleResults.Count
    TotalBytes = [int64](($downloads | Measure-Object ExpectedBytes -Sum).Sum)
    FreeBytesAfterDownload = [int64](Get-PSDrive -Name $driveName).Free
}
$summary | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $manifestDir "download_summary.json")
$summary | Format-List
