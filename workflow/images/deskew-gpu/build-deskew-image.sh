#!/usr/bin/env bash
set -euo pipefail

readonly REGISTRY=${REGISTRY:-git.biohpc.swmed.edu:5050/dean-lab}
readonly IMAGE_NAME=${IMAGE_NAME:-ctaslm2-deskew}
readonly TAG=${TAG:-0.1.0}
readonly REGISTRY_HOST=${REGISTRY%%/*}
readonly IMAGE_ROOT=$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
readonly -a PODMAN_GLOBAL_ARGS=(--cgroup-manager=cgroupfs --events-backend=file)
readonly IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

if [[ -z ${REGISTRY_USERNAME:-} || -z ${REGISTRY_PASSWORD:-} ]]; then
  echo "REGISTRY_USERNAME and REGISTRY_PASSWORD are required to publish ${IMAGE}" >&2
  exit 2
fi

printf '%s' "${REGISTRY_PASSWORD}" | podman "${PODMAN_GLOBAL_ARGS[@]}" login "${REGISTRY_HOST}" --username "${REGISTRY_USERNAME}" --password-stdin

podman "${PODMAN_GLOBAL_ARGS[@]}" build --tag "${IMAGE}" "${IMAGE_ROOT}"
podman "${PODMAN_GLOBAL_ARGS[@]}" push "${IMAGE}"

echo "Pushed ${IMAGE}"
echo "Astrocyte container URI: docker://${IMAGE}"

if command -v singularity >/dev/null 2>&1; then
  export SINGULARITY_DOCKER_USERNAME="${REGISTRY_USERNAME}"
  export SINGULARITY_DOCKER_PASSWORD="${REGISTRY_PASSWORD}"
  singularity exec "docker://${IMAGE}" python -c 'import numpy, numba, zarr, tifffile, aicsimageio, nd2, readlif; print("deskew image imports ok")'
fi
