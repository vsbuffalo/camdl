#!/usr/bin/env bash
# camdl setup script: install OCaml + Rust toolchains (if missing),
# then install OCaml deps and build the project.
#
# Supports Linux and macOS. Idempotent — safe to re-run.

set -euo pipefail

OCAML_SWITCH_VERSION="${OCAML_SWITCH_VERSION:-5.2.0}"
NO_SANDBOX="${NO_SANDBOX:-0}"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

detect_os() {
    case "$(uname -s)" in
        Linux*)  OS=linux ;;
        Darwin*) OS=macos ;;
        *)       err "Unsupported OS: $(uname -s) (Linux and macOS only)" ;;
    esac
    log "Detected OS: $OS"
}

detect_linux_pm() {
    if   have apt-get; then LINUX_PM=apt
    elif have dnf;     then LINUX_PM=dnf
    elif have yum;     then LINUX_PM=yum
    elif have pacman;  then LINUX_PM=pacman
    elif have zypper;  then LINUX_PM=zypper
    else
        warn "No supported package manager (apt/dnf/yum/pacman/zypper) found."
        LINUX_PM=unknown
    fi
}

pm_install() {
    # Usage: pm_install <pkg> [pkg...]
    case "$LINUX_PM" in
        apt)    sudo apt-get update -y && sudo apt-get install -y "$@" ;;
        dnf)    sudo dnf install -y "$@" ;;
        yum)    sudo yum install -y "$@" ;;
        pacman) sudo pacman -Sy --noconfirm "$@" ;;
        zypper) sudo zypper install -y "$@" ;;
        *)      err "Cannot install $* automatically; install manually and re-run." ;;
    esac
}

ensure_homebrew() {
    if have brew; then return; fi
    log "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for this shell
    if [[ -x /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -x /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}

ensure_base_tools() {
    log "Checking base build tools (make, git, curl, python3)..."
    local missing=()
    for t in make git curl python3; do
        have "$t" || missing+=("$t")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then return; fi

    log "Installing missing base tools: ${missing[*]}"
    if [[ "$OS" == macos ]]; then
        ensure_homebrew
        # `make`, `git`, `curl` come with Xcode CLT; trigger install if absent
        if ! xcode-select -p >/dev/null 2>&1; then
            log "Installing Xcode Command Line Tools (may prompt)..."
            xcode-select --install || true
        fi
        for t in "${missing[@]}"; do
            case "$t" in
                python3) brew list python >/dev/null 2>&1 || brew install python ;;
                *)       brew list "$t"   >/dev/null 2>&1 || brew install "$t" ;;
            esac
        done
    else
        detect_linux_pm
        local pkgs=()
        for t in "${missing[@]}"; do
            case "$t" in
                python3)
                    case "$LINUX_PM" in
                        apt) pkgs+=(python3) ;;
                        *)   pkgs+=(python3) ;;
                    esac ;;
                *) pkgs+=("$t") ;;
            esac
        done
        pm_install "${pkgs[@]}"
    fi
}

ensure_opam() {
    if have opam; then
        log "opam already installed: $(opam --version)"
        return
    fi
    log "Installing opam..."
    if [[ "$OS" == macos ]]; then
        ensure_homebrew
        brew install opam
    else
        detect_linux_pm
        case "$LINUX_PM" in
            apt)
                # Ubuntu/Debian ship recent enough opam in repos for >= 22.04
                pm_install opam
                ;;
            dnf|yum|pacman|zypper)
                pm_install opam
                ;;
            *)
                log "Falling back to opam upstream installer..."
                bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"
                ;;
        esac
    fi
}

ensure_ocaml_switch() {
    log "Ensuring opam initialized and OCaml $OCAML_SWITCH_VERSION switch exists..."

    # opam init is idempotent with --reinit guard via root presence
    if [[ ! -d "${OPAMROOT:-$HOME/.opam}" ]]; then
        if [[ "$NO_SANDBOX" == "1" ]]; then
            warn "Initializing opam without sandboxing (NO_SANDBOX=1)."
            warn "Every future 'opam install' in this switch will run build"
            warn "scripts without filesystem isolation."
            opam init --bare --disable-sandboxing -y
        elif ! opam init --bare -y; then
            cat >&2 <<'EOF'
Sandboxed opam init failed.

Most common cause on Linux: bubblewrap isn't installed,
or your kernel doesn't allow unprivileged user namespaces.

To proceed, either:
  1. Install bubblewrap and re-run:
       sudo apt-get install bubblewrap   # Debian/Ubuntu
       sudo dnf install bubblewrap       # Fedora/RHEL
       sudo pacman -S bubblewrap         # Arch
  2. Or skip sandboxing explicitly:
       NO_SANDBOX=1 ./install.sh
     (this reduces supply-chain protection on every package
     you install via opam in this switch — recommended only
     if option 1 isn't available)
EOF
            err "opam init failed without sandboxing fallback."
        fi
    fi

    # Load opam env into this shell
    eval "$(opam env --switch=default 2>/dev/null || true)"

    if ! opam switch list --short 2>/dev/null | grep -qx "$OCAML_SWITCH_VERSION"; then
        log "Creating opam switch $OCAML_SWITCH_VERSION (this can take several minutes)..."
        opam switch create "$OCAML_SWITCH_VERSION" -y
    else
        log "Switch $OCAML_SWITCH_VERSION already exists"
    fi

    opam switch set "$OCAML_SWITCH_VERSION"
    eval "$(opam env --switch="$OCAML_SWITCH_VERSION")"

    log "OCaml version: $(ocaml -version 2>&1 || echo unknown)"
}

ensure_rust() {
    if have cargo && have rustc; then
        log "Rust already installed: $(rustc --version)"
        return
    fi
    log "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    rustup default stable
}

install_ocaml_deps() {
    log "Installing OCaml package dependencies from ocaml/*.opam..."
    ( cd ocaml && opam install . --deps-only --with-test --yes )
}

build_project() {
    log "Building camdl (make build)..."
    make build
    log "Installing binaries to ~/.local/bin (make install)..."
    make install
}

verify_install() {
    log "Verifying install..."
    export PATH="$HOME/.local/bin:$PATH"
    have camdlc || err "camdlc isn't on PATH after install."
    have camdl  || err "camdl isn't on PATH after install."
    camdlc --camdl-version >/dev/null || err "camdlc was installed but won't execute."
    camdl  --version       >/dev/null || err "camdl was installed but won't execute."
    log "Verified: $(camdl --version 2>&1 | head -1)"
}

final_notes() {
    cat <<'EOF'

==========================================================================
camdl setup complete.

Next steps:
  - Ensure ~/.local/bin is on PATH:
      export PATH="$HOME/.local/bin:$PATH"
    (add to ~/.bashrc or ~/.zshrc to persist)

  - Ensure opam env loads in new shells:
      eval $(opam env)
    (add to ~/.bashrc or ~/.zshrc to persist, or run `opam init` to
     have opam wire this up automatically)

  - Verify the install:
      camdl --version
      make test
==========================================================================
EOF
}

main() {
    detect_os
    ensure_base_tools
    ensure_opam
    ensure_ocaml_switch
    ensure_rust
    install_ocaml_deps
    build_project
    verify_install
    final_notes
}

main "$@"
