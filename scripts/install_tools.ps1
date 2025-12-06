# External Tools Installation Script for Asset Generator Farm
# Run this script from PowerShell as Administrator

# ============================================================================
# CONFIGURATION
# ============================================================================

$TOOLS_DIR = "$PSScriptRoot\..\tools\external"
$VENV_DIR = "$PSScriptRoot\..\venv"

# Create tools directory
New-Item -ItemType Directory -Force -Path $TOOLS_DIR | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Asset Generator Farm - Tools Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1. AI-TOOLKIT (LoRA Training) - https://github.com/ostris/ai-toolkit
# ============================================================================
Write-Host "[1/12] Installing AI-Toolkit (LoRA Training)..." -ForegroundColor Yellow

$AI_TOOLKIT_DIR = "$TOOLS_DIR\ai-toolkit"
if (-Not (Test-Path $AI_TOOLKIT_DIR)) {
    git clone https://github.com/ostris/ai-toolkit.git $AI_TOOLKIT_DIR
    Set-Location $AI_TOOLKIT_DIR
    git submodule update --init --recursive
    # Install dependencies
    pip install -r requirements.txt
    Write-Host "  AI-Toolkit installed!" -ForegroundColor Green
} else {
    Write-Host "  AI-Toolkit already exists, skipping..." -ForegroundColor Gray
}
Set-Location $PSScriptRoot

# ============================================================================
# 2. GOOGLE PYTHON-GENAI - https://github.com/googleapis/python-genai
# ============================================================================
Write-Host "[2/12] Installing Google Python GenAI..." -ForegroundColor Yellow
pip install google-genai
Write-Host "  Google GenAI installed!" -ForegroundColor Green

# ============================================================================
# 3. LABELIMG - https://github.com/HumanSignal/labelImg
# ============================================================================
Write-Host "[3/12] Installing LabelImg (Image Annotation)..." -ForegroundColor Yellow
pip install labelImg
Write-Host "  LabelImg installed! Run with: labelImg" -ForegroundColor Green

# ============================================================================
# 4. GITHUB COPILOT CLI - https://github.com/github/copilot-cli
# ============================================================================
Write-Host "[4/12] Installing GitHub Copilot CLI..." -ForegroundColor Yellow
npm install -g @githubnext/github-copilot-cli
Write-Host "  Copilot CLI installed! Run: github-copilot-cli auth" -ForegroundColor Green

# ============================================================================
# 5. CHROMADB CHATBOT - https://github.com/daveshap/ChromaDB_Chatbot_Public
# ============================================================================
Write-Host "[5/12] Installing ChromaDB Chatbot dependencies..." -ForegroundColor Yellow
$CHROMADB_CHATBOT_DIR = "$TOOLS_DIR\chromadb-chatbot"
if (-Not (Test-Path $CHROMADB_CHATBOT_DIR)) {
    git clone https://github.com/daveshap/ChromaDB_Chatbot_Public.git $CHROMADB_CHATBOT_DIR
    Write-Host "  ChromaDB Chatbot cloned!" -ForegroundColor Green
} else {
    Write-Host "  ChromaDB Chatbot already exists, skipping..." -ForegroundColor Gray
}

# ============================================================================
# 6. MINIO (Object Storage) - https://github.com/minio/minio
# ============================================================================
Write-Host "[6/12] Installing MinIO Client..." -ForegroundColor Yellow
pip install minio
# Download MinIO server binary
$MINIO_DIR = "$TOOLS_DIR\minio"
New-Item -ItemType Directory -Force -Path $MINIO_DIR | Out-Null
$MINIO_EXE = "$MINIO_DIR\minio.exe"
if (-Not (Test-Path $MINIO_EXE)) {
    Write-Host "  Downloading MinIO server..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile $MINIO_EXE
    Write-Host "  MinIO downloaded!" -ForegroundColor Green
} else {
    Write-Host "  MinIO already downloaded, skipping..." -ForegroundColor Gray
}

# ============================================================================
# 7. BLEND_MY_NFTS - https://github.com/torrinworx/Blend_My_NFTs
# ============================================================================
Write-Host "[7/12] Installing Blend_My_NFTs (Blender addon)..." -ForegroundColor Yellow
$BLEND_NFT_DIR = "$TOOLS_DIR\blend-my-nfts"
if (-Not (Test-Path $BLEND_NFT_DIR)) {
    git clone https://github.com/torrinworx/Blend_My_NFTs.git $BLEND_NFT_DIR
    Write-Host "  Blend_My_NFTs cloned! Install as Blender addon." -ForegroundColor Green
} else {
    Write-Host "  Blend_My_NFTs already exists, skipping..." -ForegroundColor Gray
}

# ============================================================================
# 8. SKYVERN (AI Browser Automation) - https://github.com/Skyvern-AI/skyvern
# ============================================================================
Write-Host "[8/12] Installing Skyvern (AI Browser Automation)..." -ForegroundColor Yellow
pip install skyvern
Write-Host "  Skyvern installed!" -ForegroundColor Green

# ============================================================================
# 9. MCP-CHROME - https://github.com/hangwin/mcp-chrome
# ============================================================================
Write-Host "[9/12] Installing MCP-Chrome..." -ForegroundColor Yellow
$MCP_CHROME_DIR = "$TOOLS_DIR\mcp-chrome"
if (-Not (Test-Path $MCP_CHROME_DIR)) {
    git clone https://github.com/hangwin/mcp-chrome.git $MCP_CHROME_DIR
    Set-Location $MCP_CHROME_DIR
    npm install
    Write-Host "  MCP-Chrome installed!" -ForegroundColor Green
} else {
    Write-Host "  MCP-Chrome already exists, skipping..." -ForegroundColor Gray
}
Set-Location $PSScriptRoot

# ============================================================================
# 10. LANGGRAPH-MCP - https://github.com/esxr/langgraph-mcp
# ============================================================================
Write-Host "[10/12] Installing LangGraph-MCP..." -ForegroundColor Yellow
$LANGGRAPH_MCP_DIR = "$TOOLS_DIR\langgraph-mcp"
if (-Not (Test-Path $LANGGRAPH_MCP_DIR)) {
    git clone https://github.com/esxr/langgraph-mcp.git $LANGGRAPH_MCP_DIR
    Set-Location $LANGGRAPH_MCP_DIR
    pip install -r requirements.txt 2>$null
    Write-Host "  LangGraph-MCP installed!" -ForegroundColor Green
} else {
    Write-Host "  LangGraph-MCP already exists, skipping..." -ForegroundColor Gray
}
Set-Location $PSScriptRoot

# ============================================================================
# 11. SURFSENSE - https://github.com/MODSetter/SurfSense
# ============================================================================
Write-Host "[11/12] Installing SurfSense..." -ForegroundColor Yellow
$SURFSENSE_DIR = "$TOOLS_DIR\surfsense"
if (-Not (Test-Path $SURFSENSE_DIR)) {
    git clone https://github.com/MODSetter/SurfSense.git $SURFSENSE_DIR
    Write-Host "  SurfSense cloned!" -ForegroundColor Green
} else {
    Write-Host "  SurfSense already exists, skipping..." -ForegroundColor Gray
}

# ============================================================================
# 12. GENKIT - https://github.com/gemini-cli-extensions/genkit
# ============================================================================
Write-Host "[12/12] Installing Genkit..." -ForegroundColor Yellow
npm install -g genkit-cli 2>$null
pip install genkit 2>$null
Write-Host "  Genkit installed!" -ForegroundColor Green

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installed Tools:" -ForegroundColor White
Write-Host "  1. AI-Toolkit (LoRA Training) -> $TOOLS_DIR\ai-toolkit" -ForegroundColor Gray
Write-Host "  2. Google Python GenAI -> pip package" -ForegroundColor Gray
Write-Host "  3. LabelImg -> Run: labelImg" -ForegroundColor Gray
Write-Host "  4. Copilot CLI -> Run: github-copilot-cli" -ForegroundColor Gray
Write-Host "  5. ChromaDB Chatbot -> $TOOLS_DIR\chromadb-chatbot" -ForegroundColor Gray
Write-Host "  6. MinIO -> $TOOLS_DIR\minio\minio.exe" -ForegroundColor Gray
Write-Host "  7. Blend_My_NFTs -> $TOOLS_DIR\blend-my-nfts (Blender addon)" -ForegroundColor Gray
Write-Host "  8. Skyvern -> pip package" -ForegroundColor Gray
Write-Host "  9. MCP-Chrome -> $TOOLS_DIR\mcp-chrome" -ForegroundColor Gray
Write-Host "  10. LangGraph-MCP -> $TOOLS_DIR\langgraph-mcp" -ForegroundColor Gray
Write-Host "  11. SurfSense -> $TOOLS_DIR\surfsense" -ForegroundColor Gray
Write-Host "  12. Genkit -> npm/pip package" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  - Configure API keys in .env file" -ForegroundColor Gray
Write-Host "  - Start MinIO: $TOOLS_DIR\minio\minio.exe server ./data" -ForegroundColor Gray
Write-Host "  - Auth Copilot: github-copilot-cli auth" -ForegroundColor Gray
