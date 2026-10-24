# Instalador do HTRMapper para um PC novo (Windows 10 21H2+ ou 11, placa NVIDIA).
# Rode pelo INSTALAR.bat (ele pede permissão de administrador).
#
# Pode ser rodado várias vezes: cada etapa já concluída é pulada. Se o Windows
# pedir para reiniciar (instalação do WSL) ou para criar o usuário do Ubuntu,
# faça isso e rode o INSTALAR.bat de novo.
#Requires -RunAsAdministrator
param([string]$Distro = "Ubuntu-24.04")
$ErrorActionPreference = "Stop"
# Acentos certos na saída do Linux (wsl.exe) e nas mensagens.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Warn($text) { Write-Host "AVISO: $text" -ForegroundColor Yellow }
function Stop-Install($text) { Write-Host ""; Write-Host $text -ForegroundColor Red; exit 1 }

Step "1/6 Verificando o Windows"
$build = [Environment]::OSVersion.Version.Build
if ($build -lt 19044) { Stop-Install "Windows muito antigo (build $build). É preciso Windows 10 21H2 (build 19044) ou mais novo, ou Windows 11." }
Write-Host "Windows build ${build}: ok"

Step "2/6 Verificando a placa de vídeo NVIDIA"
$gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
if (-not $gpus) {
    Warn "nenhuma placa NVIDIA encontrada. A nuvem densa do HTRMapper precisa de GPU NVIDIA."
    if ((Read-Host "Continuar mesmo assim? (s/n)") -ne "s") { exit 1 }
} else {
    $gpus | ForEach-Object { Write-Host "$($_.Name), driver $($_.DriverVersion)" }
    Write-Host "Mantenha o driver NVIDIA do Windows atualizado: é ele que dá a GPU ao WSL (não instale driver NVIDIA dentro do Linux)."
}

Step "3/6 WSL (Linux dentro do Windows)"
$wslOk = $false
try { wsl.exe --status *> $null; $wslOk = ($LASTEXITCODE -eq 0) } catch { $wslOk = $false }
if (-not $wslOk) {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    if ($cpu.VirtualizationFirmwareEnabled -eq $false) {
        Warn "a virtualização parece DESLIGADA na BIOS. Ligue 'SVM Mode' (AMD) ou 'Intel VT-x' na BIOS antes, senão o WSL dá o erro HCS_E_HYPERV_NOT_INSTALLED."
    }
    wsl.exe --install --no-distribution
    Stop-Install "WSL instalado. REINICIE o computador e rode o INSTALAR.bat de novo."
}
wsl.exe --update
Write-Host "WSL: ok"

Step "4/6 Ubuntu ($Distro)"
$installed = ((wsl.exe -l -q) -replace "`0", "") | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_.Trim() }
if ($installed -notcontains $Distro) {
    Write-Host "Instalando $Distro. Quando ele pedir, crie um usuário e uma senha (anote a senha)."
    Write-Host "Se depois aparecer o prompt do Linux (algo como usuario@PC:~$), digite exit e aperte Enter."
    wsl.exe --install -d $Distro
}
$linuxUser = ""
try { $linuxUser = ((wsl.exe -d $Distro -- whoami) | Out-String).Trim() } catch { }
if ($linuxUser -eq "" -or $linuxUser -eq "root") {
    Stop-Install "O Ubuntu ainda não está pronto (sem usuário criado). Reinicie o computador se o Windows pediu, abra o 'Ubuntu' pelo Menu Iniciar, crie o usuário e a senha, e rode o INSTALAR.bat de novo."
}
Write-Host "${Distro}: ok"

Step "5/6 HTRMapper dentro do Ubuntu (vai pedir a senha do Ubuntu)"
$src = Split-Path -Parent $PSScriptRoot
$srcWsl = (wsl.exe -d $Distro -- wslpath -a "$src").Trim()
wsl.exe -d $Distro -- bash "$srcWsl/windows/instalar_wsl.sh" --criar-atalhos
if ($LASTEXITCODE -ne 0) { Stop-Install "A instalação dentro do Ubuntu falhou (veja as mensagens acima)." }

Step "6/6 Pronto"
Write-Host "Abra pelo atalho 'HTRMapper' na Área de Trabalho."
