#!/usr/bin/env bash
# camdl setup script: user-native, no-sudo install of the OCaml + Rust
# toolchains and a from-source build.
#
# This script never calls sudo and never touches a system package manager.
# Everything it installs lands under paths you own:
#   - toolchain binaries (opam, portable cmake) → $PREFIX  (default ~/.local)
#   - OCaml switch + packages                   → $HOME/.opam
#   - Rust toolchain                            → $HOME/.cargo, $HOME/.rustup
#
# It expects a handful of base build tools (make, git, curl, tar) to already
# be present; if any are missing it tells you the one-time command to install
# them and stops, rather than running a privileged install on your behalf.
#
# Supports Linux and macOS. Idempotent — safe to re-run.

set -euo pipefail

OCAML_SWITCH_VERSION="${OCAML_SWITCH_VERSION:-5.2.0}"
NO_SANDBOX="${NO_SANDBOX:-0}"

# The vendored `nlopt` C dependency builds with CMake and needs >= this.
# Older distros (e.g. Ubuntu 18.04 ships 3.10) fall below it; when the system
# cmake is missing or too old we fetch a portable build under $PREFIX.
CMAKE_MIN="3.13"
CMAKE_VERSION="${CMAKE_VERSION:-3.30.5}"

# gh#77. Install prefix — binaries land at $PREFIX/bin. Default is
# ~/.local (matches the original hardcoded behaviour). Override for
# per-branch testing, e.g.:
#   PREFIX=$HOME/.local-camdl-feat ./install.sh
#   PATH=$HOME/.local-camdl-feat/bin:$PATH camdl ...   # use the branch build
#   camdl ...                                          # back to the default install
PREFIX="${PREFIX:-$HOME/.local}"
INSTALL_DIR="$PREFIX/bin"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# true iff version $1 >= version $2 (semantic compare via `sort -V`).
# Feeds "$2\n$1" and asks sort -C whether that is already ascending: it is
# exactly when $2 <= $1. `sort -V -C` works on both GNU and BSD/macOS sort.
version_ge() { printf '%s\n%s\n' "$2" "$1" | sort -V -C; }

detect_os() {
    case "$(uname -s)" in
        Linux*)  OS=linux ;;
        Darwin*) OS=macos ;;
        *)       err "Unsupported OS: $(uname -s) (Linux and macOS only)" ;;
    esac
    ARCH="$(uname -m)"
    log "Detected: $OS/$ARCH"
}

# Map $OS/$ARCH to Kitware's portable-CMake release platform string.
cmake_plat() {
    case "$OS-$ARCH" in
        linux-x86_64)              echo linux-x86_64 ;;
        linux-aarch64|linux-arm64) echo linux-aarch64 ;;
        macos-*)                   echo macos-universal ;;
        *)                         return 1 ;;
    esac
}

# Base prerequisites: check only. We never install these — they are genuine
# system packages (git in particular drags in many runtime deps) and on any
# box that can build OCaml + Rust they are already present. If something is
# missing we print the exact one-time command and stop.
ensure_base_tools() {
    log "Checking base build tools (make, git, curl, tar, unzip, C compiler)..."
    local missing=()
    local t
    for t in make git curl tar unzip; do
        have "$t" || missing+=("$t")
    done
    # `unzip` above and a C compiler here are opam's own hard requirements: it
    # refuses to init without unzip, and the OCaml switch is compiled from
    # source (cargo also needs cc to link). Checking them HERE is what makes
    # the failure legible — opam's own error is "Missing dependencies", which
    # install.sh used to funnel into the bubblewrap/user-namespace hint below,
    # sending anyone who hit it after the wrong problem entirely.
    have cc || have gcc || have clang || missing+=("gcc")
    [ ${#missing[@]} -eq 0 ] && return

    warn "Missing required tools: ${missing[*]}"
    if [ "$OS" = macos ]; then
        cat >&2 <<EOF
Install the Xcode Command Line Tools (provides make, git, curl, tar):

    xcode-select --install

then re-run this script.
EOF
    else
        cat >&2 <<EOF
Install them once with your system package manager, e.g.:

    Debian/Ubuntu : sudo apt-get install -y ${missing[*]}
    Fedora/RHEL   : sudo dnf install -y ${missing[*]}
    Arch          : sudo pacman -S ${missing[*]}

then re-run this script. (This script never calls sudo itself — run the
above yourself, or ask an admin, on a box where you don't have root.)
EOF
    fi
    err "Missing prerequisites: ${missing[*]}"
}

# nlopt needs CMake >= $CMAKE_MIN. If the system cmake is good enough, use it.
# Otherwise fetch a portable Kitware build into $PREFIX (no sudo) and put it on
# PATH for this build — exactly the workaround dated no-sudo HPC boxes need.
ensure_cmake() {
    if have cmake; then
        local v
        v="$(cmake --version | awk 'NR==1{print $3}')"
        if version_ge "$v" "$CMAKE_MIN"; then
            log "cmake $v already present (>= $CMAKE_MIN)"
            return
        fi
        warn "System cmake $v is older than $CMAKE_MIN (nlopt needs >= $CMAKE_MIN)."
    else
        warn "cmake not found (nlopt needs >= $CMAKE_MIN)."
    fi

    local plat tarball url dir bin
    plat="$(cmake_plat)" || err "No portable CMake build for $OS/$ARCH; install cmake >= $CMAKE_MIN manually and re-run."
    tarball="cmake-${CMAKE_VERSION}-${plat}.tar.gz"
    url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${tarball}"
    dir="$PREFIX/opt/cmake-${CMAKE_VERSION}"

    log "Fetching portable CMake $CMAKE_VERSION ($plat) into $dir (no sudo)..."
    mkdir -p "$dir"
    curl -fL --proto '=https' --tlsv1.2 "$url" | tar -xz -C "$dir" --strip-components=1

    # Linux tarball: bin/cmake. macOS tarball: CMake.app/Contents/bin/cmake.
    if   [ -x "$dir/bin/cmake" ];                      then bin="$dir/bin"
    elif [ -x "$dir/CMake.app/Contents/bin/cmake" ];   then bin="$dir/CMake.app/Contents/bin"
    else err "Portable CMake fetch failed: no cmake binary under $dir"
    fi
    export PATH="$bin:$PATH"
    log "Using portable cmake: $(cmake --version | awk 'NR==1{print $1, $2, $3}')"
}

# opam: download the official prebuilt binary (SHA512-checked by the upstream
# installer's --download-only) and place it in $INSTALL_DIR. No package manager,
# no sudo. Everything downstream (opam init, the OCaml switch compile) is
# already user-native under $HOME/.opam.
ensure_opam() {
    if have opam; then
        log "opam already installed: $(opam --version)"
        return
    fi
    log "Installing opam (official prebuilt binary) into $INSTALL_DIR (no sudo)..."
    mkdir -p "$INSTALL_DIR"

    local tmp
    tmp="$(mktemp -d)"
    # The upstream installer writes ./opam-<ver>-<arch>-<os> into $PWD and exits.
    # `sh <(...)` needs bash's process substitution, hence the bash -c wrapper.
    ( cd "$tmp" && bash -c 'sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh) --download-only' )

    local bins=( "$tmp"/opam-* )
    if [ ! -e "${bins[0]}" ]; then
        rm -rf "$tmp"
        err "opam binary download failed (no opam-* in $tmp). See https://github.com/ocaml/opam/releases"
    fi
    install -m755 "${bins[0]}" "$INSTALL_DIR/opam"
    rm -rf "$tmp"

    export PATH="$INSTALL_DIR:$PATH"
    have opam || err "opam installed to $INSTALL_DIR but isn't executable / on PATH."
    log "opam installed: $(opam --version)"
}

ensure_rust() {
    if have cargo && have rustc; then
        log "Rust already installed: $(rustc --version)"
        return
    fi
    log "Installing Rust via rustup (no sudo)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    rustup default stable
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

install_ocaml_deps() {
    log "Installing OCaml package dependencies from ocaml/*.opam..."
    ( cd ocaml && opam install . --deps-only --with-test --yes )
}

build_project() {
    log "Building camdl (make build)..."
    make build
    log "Installing binaries to $INSTALL_DIR (make install)..."
    INSTALL_DIR="$INSTALL_DIR" make install
}

verify_install() {
    log "Verifying install..."
    export PATH="$INSTALL_DIR:$PATH"
    have camdlc || err "camdlc isn't on PATH after install."
    have camdl  || err "camdl isn't on PATH after install."
    camdlc --camdl-version >/dev/null || err "camdlc was installed but won't execute."
    camdl  --version       >/dev/null || err "camdl was installed but won't execute."
    log "Verified: $(camdl --version 2>&1 | head -1)"
}

final_notes() {
    cat <<EOF

==========================================================================
camdl setup complete.

Add these lines to your shell rc (~/.bashrc or ~/.zshrc) so new shells —
and non-interactive ones like \`ssh host 'camdl ...'\` — find everything:

    export PATH="$INSTALL_DIR:\$HOME/.cargo/bin:\$PATH"
    [ -f "\$HOME/.cargo/env" ] && . "\$HOME/.cargo/env"
    eval "\$(opam env)"

Verify the install:
    camdl --version
    make test
==========================================================================
EOF
}

main() {
    detect_os
    ensure_base_tools
    ensure_cmake
    ensure_opam
    ensure_rust
    ensure_ocaml_switch
    install_ocaml_deps
    build_project
    verify_install
    final_notes
}

# Run main only when executed directly, so the test harness can `source` this
# file and exercise the individual functions in isolation.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
    main "$@"
fi
