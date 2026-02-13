# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- dynamic LoRA serving mode: one base model process with adapter alias switching via vLLM runtime LoRA APIs
- new `lora` config schema (`LoraConfig`, `LoraBaseModelConfig`, `LoraAdapterConfig`)
- eager base-model startup for LoRA mode
- `config.lora.example.json` and README docs for LoRA mode
- `checkpoint_path` per-model config option for lazy CRIU restore on first request
- `--restore-detached` CLI flag (renamed from `--restore`) to restore a checkpoint and exit

### Fixed

- reject unsupported CLI/control/checkpoint flows in LoRA mode with explicit errors
- CRIU dump failure on systems without nftables (removed `--network-lock nftables`, standardized on iptables)
- CRIU restore not waking vLLM after restoring L2-checkpointed models (added wake_up + reload_weights sequence)

## [0.5.0](https://github.com/doublewordai/llmux/compare/v0.4.1...v0.5.0) - 2026-02-11

### Added

- add control API with manual switching mode
- add keep_images option to preserve checkpoints after restore

### Fixed

- transition to Running state before NCCL resume after restore
- CRIU checkpoint/restore reliability in Docker

## [0.4.1](https://github.com/doublewordai/llmux/compare/v0.4.0...v0.4.1) - 2026-02-10

### Fixed

- CRIU checkpoint networking and TP>1 restore ([#12](https://github.com/doublewordai/llmux/pull/12))

## [0.4.0](https://github.com/doublewordai/llmux/compare/v0.3.1...v0.4.0) - 2026-02-10

### Added

- support NCCL suspend/resume for TP>1 cuda-checkpoint ([#11](https://github.com/doublewordai/llmux/pull/11))
- add CRIU checkpoint and CUDA suspend sleep levels
- add drain-first TimeSlice scheduling policy

### Fixed

- reject TP>1 for CUDA suspend/checkpoint, discover all GPU PIDs

## [0.3.1](https://github.com/doublewordai/llmux/compare/v0.3.0...v0.3.1) - 2026-02-05

### Fixed

- clean up partially-woken models on wake failure to prevent CUDA OOM ([#6](https://github.com/doublewordai/llmux/pull/6))

### Other

- use Depot runners and build action

## [0.3.0](https://github.com/doublewordai/llmux/compare/v0.2.1...v0.3.0) - 2026-02-05

### Fixed

- use PAT for release-plz so releases trigger downstream workflows

### Other

- make vLLM logging always-on and remove redundant config fields

## [0.2.1](https://github.com/doublewordai/llmux/compare/v0.2.0...v0.2.1) - 2026-02-05

### Fixed

- use per-model sleep level instead of global policy default

## [0.2.0](https://github.com/doublewordai/llmux/compare/v0.1.0...v0.2.0) - 2026-02-05

### Fixed

- add anti-thrashing cooldown and zombie process recovery
- resolve clippy warnings for collapsible-if and redundant import
