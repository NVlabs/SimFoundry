#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# USAGE
# ==============================================================================
# # Interactive mode -- prompts for each key
# bash scripts/installation/login_services.sh

# # Interactive mode with a specific mamba env
# bash scripts/installation/login_services.sh --env-name hunyuan

# # Default mode -- reads keys from a file (no prompts)
# bash scripts/installation/login_services.sh --default

# # Default mode with custom keys file and env
# bash scripts/installation/login_services.sh --default --keys-file /path/to/my_keys.txt --env-name my_env

# # Pipeline VLM calls run on Google Cloud Vertex AI (Gemini), so gcloud is
# # authenticated by default. Pass --no-gcloud to skip it:
# bash scripts/installation/login_services.sh --default --no-gcloud
# ==============================================================================

# Error handling: Exit and print the offending line and error message on failure
error_handler() {
    local exit_code=$?
    local line_no=$1
    echo "Error occurred at line $line_no: $(sed "${line_no}q;d" "$0")"
    echo "Exit code: $exit_code"
    exit $exit_code
}
trap 'error_handler $LINENO' ERR
set -o errexit
set -o pipefail

eval "$(mamba shell hook --shell bash)"

# Get script dir
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
env_name="simfoundry"
DEFAULT=false
# Vertex AI backs every remote VLM stage, so Google Cloud login is on by default.
GCLOUD_LOGIN=true
KEYS_FILE="${SCRIPT_DIR}/api_keys.txt"

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name) env_name="$2"; shift 2 ;;
        --default) DEFAULT=true; shift ;;
        --no-gcloud) GCLOUD_LOGIN=false; shift ;;
        --keys-file) KEYS_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash login_services.sh [OPTIONS]"
            echo ""
            echo "Logs into Google Cloud, HuggingFace, WandB, Docker Hub, and nvcr.io (NGC)."
            echo "Google Cloud is included by default because Vertex AI backs the pipeline's"
            echo "VLM calls; pass --no-gcloud to skip it."
            echo ""
            echo "Options:"
            echo "  --env-name NAME     Mamba environment to activate (default: simfoundry)"
            echo "  --default           Read API keys from a file instead of prompting"
            echo "  --no-gcloud         Skip Google Cloud login (the pipeline will not run"
            echo "                      until you authenticate separately)"
            echo "  --keys-file PATH    Path to API keys file (default: scripts/installation/api_keys.txt)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "When using --default, the keys file should contain lines in KEY=VALUE format."
            echo "See api_keys.template.txt for the expected format."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ==============================================================================
# ACTIVATE MAMBA ENVIRONMENT
# ==============================================================================

echo "=== SimFoundry Service Login ==="

if [[ ! ${DEFAULT} == true ]]; then
    read -p "Enter mamba environment name to use (default: ${env_name}): " ENV_NAME
fi
ENV_NAME=${ENV_NAME:-${env_name}}

echo "Activating environment: $ENV_NAME"
mamba activate "$ENV_NAME"

# ==============================================================================
# LOAD KEYS (from file if --default, otherwise prompt)
# ==============================================================================

if [[ ${DEFAULT} == true ]]; then
    if [[ ! -f "$KEYS_FILE" ]]; then
        echo "ERROR: Keys file not found: $KEYS_FILE"
        echo "Create one from the template: cp api_keys.template.txt api_keys.txt"
        echo "Then fill in your keys."
        exit 1
    fi
    echo "Reading API keys from: $KEYS_FILE"
    # Source the keys file (KEY=VALUE format), skipping comments and blank lines
    while IFS='=' read -r key value; do
        # Skip comments and blank lines
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        # Trim whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        export "$key"="$value"
    done < "$KEYS_FILE"
fi

# ==============================================================================
# 1. HUGGING FACE LOGIN
# ==============================================================================

echo ""
echo "--- HuggingFace Login ---"

if [[ ${DEFAULT} == true ]]; then
    HF_TOKEN_VALUE="${HF_TOKEN:-}"
else
    read -p "Enter HuggingFace token (leave blank to skip): " HF_TOKEN_VALUE
fi

if [[ -n "$HF_TOKEN_VALUE" ]]; then
    hf auth login --token "$HF_TOKEN_VALUE"
    echo "Logged into HuggingFace."
else
    echo "Skipping HuggingFace login (no token provided)."
fi

# ==============================================================================
# 2. WANDB LOGIN
# ==============================================================================

echo ""
echo "--- Weights & Biases Login ---"

if [[ ${DEFAULT} == true ]]; then
    WANDB_KEY_VALUE="${WANDB_API_KEY:-}"
else
    read -p "Enter WandB API key (leave blank to skip): " WANDB_KEY_VALUE
fi

if [[ -n "$WANDB_KEY_VALUE" ]]; then
    wandb login "$WANDB_KEY_VALUE"
    echo "Logged into WandB."
else
    echo "Skipping WandB login (no API key provided)."
fi

# ==============================================================================
# 3. DOCKER HUB LOGIN
# ==============================================================================

echo ""
echo "--- Docker Hub Login ---"

if [[ ${DEFAULT} == true ]]; then
    DOCKER_USER_VALUE="${DOCKER_USERNAME:-}"
    DOCKER_PASS_VALUE="${DOCKER_PASSWORD:-}"
else
    read -p "Enter Docker Hub username (leave blank to skip): " DOCKER_USER_VALUE
    if [[ -n "$DOCKER_USER_VALUE" ]]; then
        read -s -p "Enter Docker Hub password/token: " DOCKER_PASS_VALUE
        echo ""
    fi
fi

if [[ -n "$DOCKER_USER_VALUE" && -n "$DOCKER_PASS_VALUE" ]]; then
    echo "$DOCKER_PASS_VALUE" | docker login -u "$DOCKER_USER_VALUE" --password-stdin
    echo "Logged into Docker Hub."
else
    echo "Skipping Docker Hub login (no credentials provided)."
fi

# ==============================================================================
# 4. NVCR.IO (NGC) LOGIN
# ==============================================================================

echo ""
echo "--- NVIDIA NGC (nvcr.io) Login ---"

if [[ ${DEFAULT} == true ]]; then
    NGC_KEY_VALUE="${NGC_API_KEY:-}"
else
    read -p "Enter NGC API key (leave blank to skip): " NGC_KEY_VALUE
fi

if [[ -n "$NGC_KEY_VALUE" ]]; then
    echo "$NGC_KEY_VALUE" | docker login nvcr.io -u '$oauthtoken' --password-stdin
    echo "Logged into nvcr.io (NGC)."
else
    echo "Skipping nvcr.io login (no API key provided)."
fi

# ==============================================================================
# 5. GOOGLE CLOUD LOGIN
# ==============================================================================

echo ""
echo "--- Google Cloud Login ---"

if [[ ${GCLOUD_LOGIN} != true ]]; then
    echo "Skipping Google Cloud login at your request (--no-gcloud)."
    echo "WARNING: Vertex AI (Gemini) backs every remote VLM stage. The pipeline will fail"
    echo "         until you run 'gcloud auth application-default login' and set a project."
elif ! command -v gcloud >/dev/null 2>&1; then
    echo "WARNING: 'gcloud' is not on PATH, so Google Cloud login was skipped." >&2
    echo "         Install the Google Cloud SDK, then re-run this script." >&2
else
    # Project id: explicit env/prompt first, then whatever gcloud is already configured with.
    if [[ ${DEFAULT} == true ]]; then
        GCLOUD_PROJECT_VALUE="${GCLOUD_PROJECT:-}"
    else
        read -p "Enter Google Cloud project ID (blank to keep the current gcloud project): " GCLOUD_PROJECT_VALUE
    fi
    if [[ -z "$GCLOUD_PROJECT_VALUE" ]]; then
        GCLOUD_PROJECT_VALUE="$(gcloud config get-value project 2>/dev/null || true)"
        [[ "$GCLOUD_PROJECT_VALUE" == "(unset)" ]] && GCLOUD_PROJECT_VALUE=""
        [[ -n "$GCLOUD_PROJECT_VALUE" ]] && echo "Using project already configured in gcloud: ${GCLOUD_PROJECT_VALUE}"
    fi

    if [[ -n "$GCLOUD_PROJECT_VALUE" ]]; then
        echo "Setting Google Cloud project: $GCLOUD_PROJECT_VALUE"
        gcloud config set project "$GCLOUD_PROJECT_VALUE"
    else
        echo "WARNING: no Google Cloud project id available." >&2
        echo "         Set GCLOUD_PROJECT, or set gcloud_project in scripts/cfg/real2sim_cfg.yaml." >&2
    fi

    # Application-default credentials are what the Vertex AI SDK actually reads.
    if gcloud auth application-default print-access-token >/dev/null 2>&1; then
        echo "Google Cloud application-default credentials already present; skipping browser login."
    else
        echo "Launching Google Cloud application-default login (this will open a browser)..."
        gcloud auth application-default login
    fi
    echo "Logged into Google Cloud."
fi

# ==============================================================================
# DONE
# ==============================================================================

echo ""
echo "=== Service login complete ==="

mamba deactivate
