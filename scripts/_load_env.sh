# Source from scripts after ROOT is set.
if [ -n "${ROOT:-}" ] && [ -f "${ROOT}/configs/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/configs/.env"
  set +a
fi
