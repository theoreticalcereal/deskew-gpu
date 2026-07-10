#!/usr/bin/env bash
set -euo pipefail

readonly REGISTRY=${REGISTRY:-git.biohpc.swmed.edu:5050/dean-lab}
readonly IMAGE_NAME=${IMAGE_NAME:-ctaslm2-deskew}
readonly TAG=${TAG:-0.1.0}
readonly REGISTRY_HOST=${REGISTRY%%/*}
readonly IMAGE_ROOT=$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
readonly -a PODMAN_GLOBAL_ARGS=(--cgroup-manager=cgroupfs --events-backend=file)
readonly IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
BUILD_ARGS=(--tag "${IMAGE}")

if [[ ${NO_CACHE:-0} == 1 || ${NO_CACHE:-false} == true ]]; then
  BUILD_ARGS=(--no-cache "${BUILD_ARGS[@]}")
fi

# Default published image: git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0

if [[ -z ${REGISTRY_USERNAME:-} || -z ${REGISTRY_PASSWORD:-} ]]; then
  echo "REGISTRY_USERNAME and REGISTRY_PASSWORD are required to publish ${IMAGE}" >&2
  exit 2
fi

printf '%s' "${REGISTRY_PASSWORD}" | podman "${PODMAN_GLOBAL_ARGS[@]}" login "${REGISTRY_HOST}" --username "${REGISTRY_USERNAME}" --password-stdin

podman "${PODMAN_GLOBAL_ARGS[@]}" build "${BUILD_ARGS[@]}" "${IMAGE_ROOT}"
podman "${PODMAN_GLOBAL_ARGS[@]}" push "${IMAGE}"

echo "Pushed ${IMAGE}"
echo "Astrocyte container URI: docker://${IMAGE}"

if command -v singularity >/dev/null 2>&1; then
  export SINGULARITY_DOCKER_USERNAME="${REGISTRY_USERNAME}"
  export SINGULARITY_DOCKER_PASSWORD="${REGISTRY_PASSWORD}"
  if id -un >/dev/null 2>&1; then
    singularity exec --nv "docker://${IMAGE}" sh -lc '
      export PATH=/opt/conda/envs/app/bin:$PATH
      python -c "import numpy, numba, zarr, tifffile, aicsimageio, nd2, readlif, pycudadecon; print(\"deskew image imports ok\")"
    '
  else
    echo "WARNING: pushed ${IMAGE}, but skipped local Singularity verification because the current UID is not resolvable." >&2
  fi
fi
