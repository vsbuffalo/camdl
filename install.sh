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
# It expects a handful of base build tools (make, git, curl, tar, unzip, and
# working C and C++ compilers) to already
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
# A compiler that exists is not the same as a compiler that works. On
# Debian/Ubuntu the `gcc` package only *recommends* libc6-dev, so a minimal
# box — or any image built with `--no-install-recommends gcc` — has `cc` on
# PATH and no crt objects to link against. `command -v cc` passes; the failure
# then resurfaces as "C compiler cannot create executables" inside the OCaml
# switch build, hundreds of lines later, which is precisely the kind of
# late-and-misattributed report this preflight exists to prevent (gh#755).
# So compile and link something trivial and see.
#
# $1 is the driver to invoke, $2 the source extension to hand it.
compiler_works() {
    local drv="$1" ext="$2"
    have "$drv" || return 1
    local d rc=0
    d="$(mktemp -d)" || return 1
    printf 'int main(void){return 0;}\n' > "$d/probe.$ext"
    "$drv" -o "$d/probe" "$d/probe.$ext" >/dev/null 2>&1 || rc=1
    rm -rf "$d"
    return "$rc"
}

# Several probes can map to one package — build-essential covers both cc and
# c++ — and printing it twice makes the install line look like a mistake.
dedupe() { printf '%s\n' "$@" | awk '!seen[$0]++' | tr '\n' ' ' | sed 's/ $//'; }

ensure_base_tools() {
    log "Checking base build tools (make, git, curl, tar, unzip, cc, c++)..."
    # The probe list is what the rest of this script cannot proceed without:
    #   make git curl tar  the build itself
    #   unzip              opam refuses to init without it — a hard
    #                      requirement, not a nicety
    #   cc                 the OCaml switch is compiled from source, and
    #                      cargo invokes `cc` as its linker driver
    #   c++                the vendored nlopt declares a CXX CMake project, so
    #                      its build configures for a C++ compiler even though
    #                      camdl links only the C library
    #
    # Probing a command and naming a package are different things, and for the
    # compiler they are different per distro too: no distro ships a package
    # called `cc`, Debian/Ubuntu need build-essential to get a compiler that
    # can actually link, and gcc alone is right on Fedora and Arch. Printing
    # the probe name, or one package name for everyone, hands the reader an
    # install line that does not work.
    local missing=() apt_pkgs=() dnf_pkgs=() pac_pkgs=()
    local t
    for t in make git curl tar unzip cc c++; do
        case "$t" in
            cc)  compiler_works cc  c  && continue ;;
            c++) compiler_works c++ cc && continue ;;
            *)   have "$t" && continue ;;
        esac
        missing+=("$t")
        case "$t" in
            cc)  apt_pkgs+=(build-essential); dnf_pkgs+=(gcc);     pac_pkgs+=(gcc) ;;
            c++) apt_pkgs+=(build-essential); dnf_pkgs+=(gcc-c++); pac_pkgs+=(gcc) ;;
            *)   apt_pkgs+=("$t"); dnf_pkgs+=("$t"); pac_pkgs+=("$t") ;;
        esac
    done
    [ ${#missing[@]} -eq 0 ] && return

    warn "Missing or unusable required tools: ${missing[*]}"
    if [ "$OS" = macos ]; then
        cat >&2 <<EOF
Install the Xcode Command Line Tools (provides make, git, curl, tar, cc, c++;
unzip ships with macOS):

    xcode-select --install

then re-run this script.
EOF
    else
        cat >&2 <<EOF
Install them once with your system package manager, e.g.:

    Debian/Ubuntu : sudo apt-get install -y $(dedupe "${apt_pkgs[@]}")
    Fedora/RHEL   : sudo dnf install -y $(dedupe "${dnf_pkgs[@]}")
    Arch          : sudo pacman -S $(dedupe "${pac_pkgs[@]}")

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
            # Report what was checked, not a guess. This branch used to assert
            # that bubblewrap was probably missing or that the kernel blocked
            # unprivileged user namespaces — for *any* opam init failure. On a
            # box where bubblewrap was installed and namespaces were fine, that
            # sent the reader off to debug a problem that did not exist, while
            # the real cause sat in opam's own output just above (gh#755).
            echo >&2
            if [ "$OS" != macos ] && ! have bwrap; then
                cat >&2 <<'EOF'
opam init failed, and bubblewrap (bwrap) is not installed. opam sandboxes
every package build with it, so that is the likely cause:

    sudo apt-get install bubblewrap   # Debian/Ubuntu
    sudo dnf install bubblewrap       # Fedora/RHEL
    sudo pacman -S bubblewrap         # Arch
EOF
            else
                cat >&2 <<'EOF'
opam init failed. Its own output above is the authoritative reason; read
that first. If it names bwrap, a namespace, or a permission error, then
the sandbox is being blocked by your kernel or container runtime rather
than by a missing package.
EOF
            fi
            cat >&2 <<'EOF'

If the sandbox cannot be made to work, skip it explicitly:

    NO_SANDBOX=1 ./install.sh

That reduces supply-chain protection on every package you install via opam
in this switch, so prefer fixing the sandbox where you can.
EOF
            err "opam init failed."
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
