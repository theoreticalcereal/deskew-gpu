#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_IMAGE=${SOURCE_IMAGE:-git.biohpc.swmed.edu:5050/dean-lab/neuroglancer/deskew-gpu:0.1.0}
readonly TARGET_IMAGE=${TARGET_IMAGE:-git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0}
readonly REGISTRY_HOST=${TARGET_IMAGE%%/*}
readonly -a PODMAN_GLOBAL_ARGS=(--cgroup-manager=cgroupfs --events-backend=file)

if [[ -z ${REGISTRY_USERNAME:-} || -z ${REGISTRY_PASSWORD:-} ]]; then
  echo "REGISTRY_USERNAME and REGISTRY_PASSWORD are required to retag ${SOURCE_IMAGE}" >&2
  exit 2
fi

printf '%s' "${REGISTRY_PASSWORD}" | podman "${PODMAN_GLOBAL_ARGS[@]}" login "${REGISTRY_HOST}" --username "${REGISTRY_USERNAME}" --password-stdin

podman "${PODMAN_GLOBAL_ARGS[@]}" pull "${SOURCE_IMAGE}"
podman "${PODMAN_GLOBAL_ARGS[@]}" tag "${SOURCE_IMAGE}" "${TARGET_IMAGE}"
if ! podman "${PODMAN_GLOBAL_ARGS[@]}" push "${TARGET_IMAGE}"; then
  cat >&2 <<EOF

ERROR: failed to push ${TARGET_IMAGE}.

The local image was pulled and retagged successfully, so this is a registry
write/permission problem for the target repository. Confirm that:

  1. git.biohpc.swmed.edu/dean-lab/ctaslm2-deskew exists.
  2. Its container registry is enabled.
  3. REGISTRY_USERNAME has Developer/Maintainer access or a token with write_registry.

After fixing permissions, rerun:
  SOURCE_IMAGE=${SOURCE_IMAGE} TARGET_IMAGE=${TARGET_IMAGE} $0
EOF
  exit 1
fi

echo "Copied ${SOURCE_IMAGE}"
echo "    to ${TARGET_IMAGE}"
echo "Astrocyte container URI: docker://${TARGET_IMAGE}"
