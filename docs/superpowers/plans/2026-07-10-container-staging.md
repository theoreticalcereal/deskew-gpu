# Container Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deskew GPU always runs from `git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0`.

**Architecture:** The Astrocyte manifest declares the required Singularity image, and each Nextflow process that runs Python declares the same container. The conda runtime builder, runtime reuse, and runtime export path are removed from the active workflow.

**Tech Stack:** Nextflow DSL2, Singularity, Astrocyte package metadata, pytest static wiring tests.

## Global Constraints

- Process container image: `git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0`
- Astrocyte staging image: `docker://git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0`
- CUDA module: `cuda/11.8.0`
- No runtime fallback to host conda.
- Keep existing input/output behavior unchanged.
- Preserve unrelated untracked files and user changes.

---

### Task 1: Replace Runtime Wiring With Process Containers

**Files:**
- Modify: `workflow/modules.nf`
- Modify: `workflow/main.nf`
- Modify: `workflow/configs/nextflow.config`
- Modify: `workflow/configs/biohpc.config`
- Delete: `workflow/envs/deskew-conda.txt`
- Delete: `workflow/envs/deskew-pip-constraints.txt`
- Delete: `workflow/envs/deskew-pip-requirements.txt`

**Interfaces:**
- Consumes: existing process names `STAGE_DESKEW_INPUT`, `DESKEW`, `EXPORT_OUTPUT_FORMAT`.
- Produces: Python processes that run inside `git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0` without a `deskew_runtime` input.

- [ ] **Step 1: Add/adjust static tests first**

Add assertions to `tests/test_deskew_wiring.py` that fail while runtime-builder wiring still exists:

```python
def test_workflow_uses_fixed_container_runtime():
    modules = (ROOT / "workflow" / "modules.nf").read_text()
    main = (ROOT / "workflow" / "main.nf").read_text()
    package = (ROOT / "astrocyte_pkg.yml").read_text()
    image = "git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0"
    staging_image = f"docker://{image}"

    assert image in modules
    assert staging_image in package
    assert "process BUILD_DESKEW_CONTAINER" not in modules
    assert "BUILD_DESKEW_CONTAINER" not in main
    assert "deskew_runtime" not in modules
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_deskew_wiring.py -q`

Expected before implementation: failure because the fixed image is not yet used and runtime wiring still exists.

- [ ] **Step 3: Update workflow implementation**

In `workflow/modules.nf`, define the fixed image once and add `container DESKEW_CONTAINER_IMAGE` to `STAGE_DESKEW_INPUT`, `DESKEW`, and `EXPORT_OUTPUT_FORMAT`. Remove `BUILD_DESKEW_CONTAINER`, all `path deskew_runtime` inputs, and shell activation blocks.

In `workflow/main.nf`, remove `BUILD_DESKEW_CONTAINER` import and channel creation. Call the three remaining processes without `deskew_container_ch`.

In `workflow/configs/nextflow.config` and `workflow/configs/biohpc.config`, remove `params.build_deskew_container`, the `conda_runtime` profile, and conda builder defaults.

Delete `workflow/envs/deskew-conda.txt`, `workflow/envs/deskew-pip-constraints.txt`, and `workflow/envs/deskew-pip-requirements.txt` because they only supported the removed manual conda builder path.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `pytest tests/test_deskew_wiring.py -q`

Expected after implementation: all tests pass.

### Task 2: Update Package Metadata And Docs

**Files:**
- Modify: `astrocyte_pkg.yml`
- Modify: `README.md`
- Modify: `docs/workflow-overview.md`
- Modify: `docs/profiles-and-parameters.md`
- Modify: `docs/outputs-and-troubleshooting.md`

**Interfaces:**
- Consumes: fixed container image from Task 1.
- Produces: package metadata and user docs that describe the fixed container runtime only.

- [ ] **Step 1: Update static tests for docs/metadata**

Extend `tests/test_deskew_wiring.py` so it asserts no runtime parameters remain in `astrocyte_pkg.yml`, `README.md`, or docs:

```python
def test_runtime_builder_docs_removed():
    checked = [
        ROOT / "astrocyte_pkg.yml",
        ROOT / "README.md",
        ROOT / "docs" / "workflow-overview.md",
        ROOT / "docs" / "profiles-and-parameters.md",
        ROOT / "docs" / "outputs-and-troubleshooting.md",
    ]
    forbidden = [
        "deskew_runtime_dir",
        "export_deskew_runtime",
        "BUILD_DESKEW_CONTAINER",
        "conda runtime",
        "built deskew runtime",
    ]

    for path in checked:
        text = path.read_text()
        for value in forbidden:
            assert value not in text, f"{value!r} remained in {path}"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_deskew_wiring.py -q`

Expected before docs updates: failure because runtime builder text remains.

- [ ] **Step 3: Update metadata and docs**

Add `workflow_modules: ['singularity/3.9.9']` semantics to match the fixed container path and add `workflow_containers` with the image. Remove conda/anaconda workflow module dependency and remove runtime parameters. Update docs to describe the fixed BioHPC container and delete runtime export/reuse instructions.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_deskew_wiring.py -q`

Expected: all tests pass.

### Task 3: Repository Verification

**Files:**
- Verify: all modified workflow, docs, and tests.

**Interfaces:**
- Consumes: Task 1 and Task 2 outputs.
- Produces: verified repo state.

- [ ] **Step 1: Run static search**

Run: `rg -n "BUILD_DESKEW_CONTAINER|deskew_runtime_dir|export_deskew_runtime|build_deskew_container|conda_runtime|deskew_runtime" workflow astrocyte_pkg.yml README.md docs tests`

Expected: no active workflow/package/doc references except historical notes in the new spec/plan if included in the search.

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`

Expected: pass.

- [ ] **Step 3: Run Nextflow syntax/help check if available**

Run: `nextflow -version`

If Nextflow exists, run: `cd workflow && nextflow run main.nf -stub-run --input ../test_data/astrocyte_dummy.ozx --dx 0.168 --dz 0.2 --angle 45 --deskew_backend cpu_blocked`

Expected: command starts workflow planning without conda solve or runtime build. If Nextflow is unavailable or cluster submission blocks local execution, record that limitation.
