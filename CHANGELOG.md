# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.10](https://github.com/doublewordai/llmux/compare/v0.7.9...v0.7.10) - 2026-02-20

### Fixed

- let CRIU CUDA plugin handle GPU toggle, clean /dev/shm

## [0.7.9](https://github.com/doublewordai/llmux/compare/v0.7.8...v0.7.9) - 2026-02-20

### Fixed

- include parent process in CUDA context discovery

## [0.7.8](https://github.com/doublewordai/llmux/compare/v0.7.7...v0.7.8) - 2026-02-20

### Fixed

- discover all CUDA-holding processes for checkpoint pre-toggle

## [0.7.7](https://github.com/doublewordai/llmux/compare/v0.7.6...v0.7.7) - 2026-02-20

### Fixed

- restore cuda-checkpoint pre-toggle before CRIU dump

## [0.7.6](https://github.com/doublewordai/llmux/compare/v0.7.5...v0.7.6) - 2026-02-19

### Fixed

- increase startup timeout to 30 minutes for large MoE models

## [0.7.5](https://github.com/doublewordai/llmux/compare/v0.7.4...v0.7.5) - 2026-02-19

### Fixed

- regenerate reload_weights patch with correct v0.15.1 line numbers

## [0.7.4](https://github.com/doublewordai/llmux/compare/v0.7.3...v0.7.4) - 2026-02-19

### Added

- checkpoint reuse, stray GPU cleanup, and reload_weights patch ([#25](https://github.com/doublewordai/llmux/pull/25))

## [0.7.3](https://github.com/doublewordai/llmux/compare/v0.7.2...v0.7.3) - 2026-02-19

### Fixed

- use both --link-remap and --ghost-limit for CRIU dumps ([#23](https://github.com/doublewordai/llmux/pull/23))

## [0.7.2](https://github.com/doublewordai/llmux/compare/v0.7.1...v0.7.2) - 2026-02-19

### Fixed

- use ghost files instead of link-remap for CRIU checkpoints ([#21](https://github.com/doublewordai/llmux/pull/21))

## [0.7.1](https://github.com/doublewordai/llmux/compare/v0.7.0...v0.7.1) - 2026-02-19

### Fixed

- always checkpoint outgoing model during switch ([#19](https://github.com/doublewordai/llmux/pull/19))

## [0.7.0](https://github.com/doublewordai/llmux/compare/v0.6.0...v0.7.0) - 2026-02-19

### Added

- add toggleable warmup phase

## [0.6.0](https://github.com/doublewordai/llmux/compare/v0.5.0...v0.6.0) - 2026-02-12

### Added

- add S3 object store for CRIU checkpoint persistence
- add --checkpoint and --restore CLI commands

### Other

- rewrite eviction policy section with interaction matrix
- replace --levels with --policies in validate CLI
- remove old L1-L5 references from README
- update README for two-axis eviction policy
- replace SleepLevel with two-axis EvictionPolicy (weights + process)

### Added

- `checkpoint_path` per-model config option for lazy CRIU restore on first request
- `--restore-detached` CLI flag (renamed from `--restore`) to restore a checkpoint and exit

### Fixed

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
