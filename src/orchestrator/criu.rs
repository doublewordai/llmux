use super::{
    ManagedProcess, Orchestrator, OrchestratorError, ProcessState, kill_process_group, maybe_sudo,
};
use crate::types::{EvictionPolicy, WeightStrategy};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

impl Orchestrator {
    /// Clean stale /dev/shm entries that interfere with CRIU dumps.
    fn clean_devshm() {
        let shm = Path::new("/dev/shm");
        if let Ok(entries) = std::fs::read_dir(shm) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("link_remap.") || name.starts_with("sem.mp-") {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    /// Directories in the container's ephemeral filesystem that vLLM creates at
    /// runtime. These don't exist in a fresh container, so CRIU restore fails if
    /// they're not recreated. We save them into the checkpoint directory during
    /// dump and restore them before CRIU restore.
    const EPHEMERAL_DIRS: &'static [&'static str] = &[
        "/root/.cache/flashinfer",
        "/root/.cache/tvm-ffi",
        "/root/.triton/cache",
    ];

    /// Save runtime-generated files from the container's ephemeral filesystem
    /// into the checkpoint directory. Called after CRIU dump (the files are still
    /// on disk even though the process is killed).
    fn save_ephemeral_files(images_dir: &Path) {
        let rootfs_dir = images_dir.join("rootfs");
        for dir in Self::EPHEMERAL_DIRS {
            let src = Path::new(dir);
            if !src.exists() {
                continue;
            }
            let dest = rootfs_dir.join(dir.trim_start_matches('/'));
            if let Err(e) = Self::copy_dir_recursive(src, &dest) {
                warn!(src = %src.display(), error = %e, "Failed to save ephemeral dir (non-fatal)");
            } else {
                info!(src = %src.display(), dest = %dest.display(), "Saved ephemeral dir to checkpoint");
            }
        }
    }

    /// Restore runtime-generated files from the checkpoint directory back into
    /// the container's filesystem. Called before CRIU restore.
    fn restore_ephemeral_files(images_dir: &Path) {
        let rootfs_dir = images_dir.join("rootfs");
        if !rootfs_dir.exists() {
            return;
        }
        for dir in Self::EPHEMERAL_DIRS {
            let src = rootfs_dir.join(dir.trim_start_matches('/'));
            let dest = Path::new(dir);
            if !src.exists() {
                continue;
            }
            if let Err(e) = Self::copy_dir_recursive(&src, dest) {
                warn!(src = %src.display(), dest = %dest.display(), error = %e,
                      "Failed to restore ephemeral dir (non-fatal)");
            } else {
                info!(src = %src.display(), dest = %dest.display(), "Restored ephemeral dir from checkpoint");
            }
        }
    }

    /// Recursively copy a directory tree.
    fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dest)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dest_path = dest.join(entry.file_name());
            if src_path.is_dir() {
                Self::copy_dir_recursive(&src_path, &dest_path)?;
            } else {
                std::fs::copy(&src_path, &dest_path)?;
            }
        }
        Ok(())
    }

    /// Checkpoint a model to disk using CRIU with the CUDA plugin.
    ///
    /// 1. Clean stale /dev/shm entries
    /// 2. Dump the process tree via CRIU (writes everything to disk, kills process)
    /// 3. Update state to Checkpointed
    ///
    /// When `keep_images` is true and a valid checkpoint already exists on disk,
    /// the CRIU dump is skipped — the process is simply killed and the existing
    /// images are reused. This is safe because there's no meaningful runtime
    /// state to preserve (KV cache is discarded during sleep anyway).
    pub(super) async fn checkpoint_model(
        &self,
        model: &str,
        process: &Arc<Mutex<ManagedProcess>>,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        let ckpt_cfg =
            self.checkpoint_config
                .as_ref()
                .ok_or_else(|| OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "Checkpoint level requested but no checkpoint config".to_string(),
                })?;

        let parent_pid = {
            let guard = process.lock().await;
            guard
                .child
                .as_ref()
                .and_then(|c| c.id())
                .or(guard.parent_pid)
                .ok_or_else(|| OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "No child PID available for checkpoint".to_string(),
                })?
        };

        let images_dir = ckpt_cfg.images_dir.join(model).join("images");

        // Reuse existing checkpoint: if keep_images is true and a valid
        // checkpoint already exists, skip the expensive CRIU dump and just
        // kill the process. The existing images will be used for restore.
        let has_existing_checkpoint = images_dir.exists()
            && std::fs::read_dir(&images_dir)
                .map(|mut d| d.next().is_some())
                .unwrap_or(false);

        if ckpt_cfg.keep_images && has_existing_checkpoint {
            info!(
                model = %model,
                images_dir = %images_dir.display(),
                "Reusing existing checkpoint images, killing process instead of re-checkpointing"
            );

            // Kill the process group to free GPU memory
            let mut guard = process.lock().await;
            if let Some(ref mut child) = guard.child {
                if let Some(pid) = child.id() {
                    kill_process_group(pid);
                } else {
                    let _ = child.kill().await;
                }
                let _ = child.wait().await;
            }
            guard.child = None;
            guard.state = ProcessState::Checkpointed {
                images_dir: images_dir.clone(),
                eviction,
            };

            return Ok(());
        }

        // Prepare images directory (clean up old checkpoint first)
        if images_dir.exists() {
            std::fs::remove_dir_all(&images_dir).map_err(|e| OrchestratorError::SleepFailed {
                model: model.to_string(),
                reason: format!("Failed to clean old checkpoint: {}", e),
            })?;
        }
        std::fs::create_dir_all(&images_dir).map_err(|e| OrchestratorError::SleepFailed {
            model: model.to_string(),
            reason: format!("Failed to create images dir: {}", e),
        })?;

        // Clean stale /dev/shm entries from previous CRIU dumps.
        // CRIU --link-remap creates hardlinks (link_remap.N) for deleted-but-open
        // files. If these accumulate, subsequent dumps fail with "File exists".
        // Python multiprocessing semaphores (sem.mp-*) from killed processes also
        // linger. Cleaning these before dump prevents conflicts.
        Self::clean_devshm();

        // Save CUDA PIDs so restore can track the process after
        // --restore-detached (engine_core_pids is empty in our state,
        // but the processes run at their original PIDs).
        {
            let guard = process.lock().await;
            let pids: Vec<String> = guard
                .engine_core_pids
                .iter()
                .map(|p| p.to_string())
                .collect();
            let pids_file = images_dir.join("cuda_pids");
            if let Err(e) = std::fs::write(&pids_file, pids.join("\n")) {
                warn!(model = %model, error = %e, "Failed to save CUDA PIDs (non-fatal)");
            }
        }

        // CRIU dump (snapshots process tree to disk, kills it).
        //
        // The CRIU CUDA plugin (-L) handles GPU device file descriptors
        // and the CUDA driver's kernel-level state during dump/restore.
        // For retain strategy, CUDA was pre-suspended via --toggle above.
        // For offload/discard, vLLM sleep quiesced CUDA enough for the plugin.
        //
        // --link-remap is required for CRIU to handle deleted-but-open files
        // (e.g. Python multiprocessing semaphores in /dev/shm that are
        // sem_unlink'd while the fd is still open). During dump, CRIU creates
        // temporary hardlinks (link_remap.N) to preserve these inodes.
        //
        // --ghost-limit embeds deleted file contents directly in the checkpoint
        // image. This makes restores self-contained: even if the link_remap
        // hardlinks are destroyed (e.g. by a different model's checkpoint),
        // CRIU can still restore from the embedded ghost file.
        info!(model = %model, parent_pid, "Checkpointing: criu dump");
        let criu_dump = maybe_sudo(&ckpt_cfg.criu_path)
            .args([
                "dump",
                "--shell-job",
                "--ext-unix-sk",
                "--tcp-established",
                "--link-remap",
                "--ghost-limit",
                "1048576",
                "--enable-external-masters",
                "-L",
                &ckpt_cfg.cuda_plugin_dir,
                "--images-dir",
                &images_dir.to_string_lossy(),
                "--tree",
                &parent_pid.to_string(),
            ])
            .output()
            .await
            .map_err(|e| OrchestratorError::SleepFailed {
                model: model.to_string(),
                reason: format!("Failed to run criu dump: {}", e),
            })?;

        if !criu_dump.status.success() {
            let stderr = String::from_utf8_lossy(&criu_dump.stderr);
            error!(model = %model, "criu dump failed: {}", stderr);
            return Err(OrchestratorError::SleepFailed {
                model: model.to_string(),
                reason: format!("criu dump failed: {}", stderr),
            });
        }

        // CRIU dump kills the process after snapshotting, so clean up our handle
        {
            let mut guard = process.lock().await;
            // The process is gone — try_wait to reap the zombie
            if let Some(ref mut child) = guard.child {
                let _ = child.try_wait();
            }
            guard.child = None;
            guard.state = ProcessState::Checkpointed {
                images_dir: images_dir.clone(),
                eviction,
            };
        }

        // Save the parent PID so cross-container restore can track the process.
        // After CRIU --restore-detached, there's no child handle, but the process
        // runs at its original PID. We need this for the stray GPU detector.
        let pid_file = images_dir.join("parent_pid");
        if let Err(e) = std::fs::write(&pid_file, parent_pid.to_string()) {
            warn!(model = %model, error = %e, "Failed to save parent PID (non-fatal)");
        }

        info!(model = %model, images_dir = %images_dir.display(), "Model checkpointed to disk");

        // Save runtime-generated files so cross-container restore works.
        // These files live in the container's ephemeral OverlayFS and won't
        // exist in a new container.
        Self::save_ephemeral_files(&images_dir);

        // Upload to S3 if object store is configured
        if let Some(ref obj_cfg) = ckpt_cfg.object_store {
            match crate::object_store::CheckpointStore::new(obj_cfg) {
                Ok(store) => {
                    info!(model = %model, "Uploading checkpoint to object store");
                    if let Err(e) = store.upload_checkpoint(model, &images_dir).await {
                        warn!(model = %model, error = %e,
                              "Failed to upload checkpoint to object store (local copy preserved)");
                    }
                }
                Err(e) => {
                    warn!(model = %model, error = %e, "Failed to initialize object store client");
                }
            }
        }

        Ok(())
    }

    /// Restore a model from a CRIU checkpoint.
    ///
    /// 1. Run criu restore (process comes back with original PID, all CUDA state intact)
    /// 2. Health-check the vLLM endpoint (should be immediately ready)
    /// 3. Update state to Running
    pub(super) async fn restore_checkpoint(
        &self,
        model: &str,
        images_dir: &Path,
        eviction: EvictionPolicy,
    ) -> Result<(), OrchestratorError> {
        let ckpt_cfg =
            self.checkpoint_config
                .as_ref()
                .ok_or_else(|| OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: "Checkpoint config missing for restore".to_string(),
                })?;

        let config = self
            .configs
            .get(model)
            .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;

        info!(model = %model, images_dir = %images_dir.display(), "Restoring from CRIU checkpoint");

        // Download from S3 if local images are missing
        let needs_download = !images_dir.exists()
            || std::fs::read_dir(images_dir)
                .map(|mut d| d.next().is_none())
                .unwrap_or(true);

        if needs_download {
            if let Some(ref obj_cfg) = ckpt_cfg.object_store {
                let store = crate::object_store::CheckpointStore::new(obj_cfg).map_err(|e| {
                    OrchestratorError::WakeFailed {
                        model: model.to_string(),
                        reason: format!("Failed to init object store: {}", e),
                    }
                })?;

                info!(model = %model, "Local checkpoint not found, downloading from S3");
                store
                    .download_checkpoint(model, images_dir)
                    .await
                    .map_err(|e| OrchestratorError::WakeFailed {
                        model: model.to_string(),
                        reason: format!("Failed to download checkpoint from S3: {}", e),
                    })?;
            } else {
                return Err(OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: format!(
                        "Checkpoint images not found at {} and no object store configured",
                        images_dir.display()
                    ),
                });
            }
        }

        // Clean stale /dev/shm entries before restore
        Self::clean_devshm();

        // Restore runtime-generated files that were saved during checkpoint.
        // These files live in the container's ephemeral OverlayFS and won't
        // exist in a fresh container (cross-container restore).
        Self::restore_ephemeral_files(images_dir);

        // Run criu restore with CUDA plugin (-L) to properly restore GPU
        // device fds. After restore, we resume CUDA via manual toggle.
        let criu_restore = maybe_sudo(&ckpt_cfg.criu_path)
            .args([
                "restore",
                "--shell-job",
                "--ext-unix-sk",
                "--tcp-established",
                "--enable-external-masters",
                "--restore-detached",
                "-L",
                &ckpt_cfg.cuda_plugin_dir,
                "--images-dir",
                &images_dir.to_string_lossy(),
            ])
            .output()
            .await
            .map_err(|e| OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("Failed to run criu restore: {}", e),
            })?;

        if !criu_restore.status.success() {
            let stderr = String::from_utf8_lossy(&criu_restore.stderr);
            error!(model = %model, "criu restore failed: {}", stderr);
            return Err(OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("criu restore failed: {}", stderr),
            });
        }

        info!(model = %model, "criu restore succeeded");

        // Restore CUDA PIDs from checkpoint (needed for process tracking
        // and for toggle resume if applicable).
        {
            let process = self
                .processes
                .get(model)
                .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
            let mut guard = process.lock().await;

            let pids_file = images_dir.join("cuda_pids");
            if let Ok(content) = std::fs::read_to_string(&pids_file) {
                let pids: Vec<u32> = content
                    .lines()
                    .filter_map(|l| l.trim().parse().ok())
                    .collect();
                if !pids.is_empty() {
                    guard.tp_size = pids.len().saturating_sub(1).max(1);
                    info!(model = %model, ?pids, tp = guard.tp_size, "Restored CUDA PIDs from checkpoint");
                    guard.engine_core_pids = pids;
                }
            }
        }

        // Health-check: the restored process should be immediately ready
        // (all state including CUDA graphs is preserved)
        let health_url = format!("http://localhost:{}/health", config.port);
        let mut ready = false;
        for attempt in 0..30 {
            match self.check_health(&health_url).await {
                Ok(true) => {
                    ready = true;
                    break;
                }
                _ => {
                    debug!(model = %model, attempt, "Post-restore health check pending...");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }

        if !ready {
            return Err(OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: "Restored process failed health check".to_string(),
            });
        }

        // For TP>1: rebuild NCCL communicators after restore
        // (they were torn down before checkpoint via suspend_nccl)
        let gpu_count = {
            let process = self
                .processes
                .get(model)
                .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
            process.lock().await.tp_size
        };
        // Update state before NCCL resume — the process is already live after
        // CRIU restore, so mark it Running so cleanup paths (e.g. force_sleep)
        // correctly kill the process rather than just deleting images.
        {
            let process = self
                .processes
                .get(model)
                .ok_or_else(|| OrchestratorError::ModelNotFound(model.to_string()))?;
            let mut guard = process.lock().await;
            guard.state = ProcessState::Running { sleeping: None };
            // After criu restore --restore-detached, there's no tokio Child handle.
            // Read the parent PID from the checkpoint so the stray GPU detector
            // knows this process is ours (not a stray to be killed).
            let pid_file = images_dir.join("parent_pid");
            if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = pid_str.trim().parse::<u32>() {
                    guard.parent_pid = Some(pid);
                    info!(model = %model, pid, "Restored parent PID from checkpoint");
                }
            }
        }

        // For TP>1: rebuild NCCL communicators after restore
        if gpu_count > 1 {
            let base_url = format!("http://localhost:{}", config.port);
            info!(model = %model, tp = gpu_count, "Resuming NCCL communicators");
            self.post_request(
                &format!("{}/collective_rpc", base_url),
                Some(r#"{"method": "resume_nccl"}"#),
                Duration::from_secs(30),
            )
            .await
            .map_err(|e| OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("Failed to resume NCCL: {}", e),
            })?;
        }

        // If vLLM sleep was used before checkpoint, run the wake sequence
        // (wake_up → reload_weights → reset_prefix_cache)
        if eviction.needs_vllm_sleep() {
            let base_url = format!("http://localhost:{}", config.port);

            info!(model = %model, "Post-restore: waking vLLM (POST /wake_up)");
            self.post_request(
                &format!("{}/wake_up", base_url),
                None,
                Duration::from_secs(30),
            )
            .await
            .map_err(|e| OrchestratorError::WakeFailed {
                model: model.to_string(),
                reason: format!("Post-restore wake_up failed: {}", e),
            })?;

            if eviction.weights == WeightStrategy::Discard {
                info!(model = %model, "Post-restore: reloading weights (POST /collective_rpc)");
                self.post_request(
                    &format!("{}/collective_rpc", base_url),
                    Some(r#"{"method": "reload_weights"}"#),
                    Duration::from_secs(60),
                )
                .await
                .map_err(|e| OrchestratorError::WakeFailed {
                    model: model.to_string(),
                    reason: format!("Post-restore reload_weights failed: {}", e),
                })?;

                info!(model = %model, "Post-restore: resetting prefix cache");
                self.post_request(
                    &format!("{}/reset_prefix_cache", base_url),
                    None,
                    Duration::from_secs(30),
                )
                .await
                .ok(); // non-fatal
            }
        }

        if !ckpt_cfg.keep_images {
            // Clean up checkpoint images to free disk space. For large models
            // this can be tens of GB.
            if let Err(e) = std::fs::remove_dir_all(images_dir) {
                warn!(model = %model, error = %e, "Failed to clean up checkpoint images (non-fatal)");
            } else {
                info!(model = %model, "Cleaned up checkpoint images");
            }
        }

        info!(model = %model, "Model restored from checkpoint and ready");
        Ok(())
    }
}
