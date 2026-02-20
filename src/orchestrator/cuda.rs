use super::{ManagedProcess, Orchestrator, OrchestratorError, maybe_sudo};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Find all descendant processes that have active CUDA contexts.
///
/// vLLM spawns a child process tree. The EngineCore parent initialises CUDA
/// before forking GPU workers. Both the parent and its worker children may
/// therefore hold CUDA contexts that must be toggled before CRIU dump.
///
/// Primary method: `cuda-checkpoint --get-state --pid <PID>` on every
/// descendant. Processes whose state is "running" have active CUDA contexts.
///
/// Fallback (if cuda-checkpoint is unavailable): check for `/dev/nvidia`
/// device mappings in `/proc/PID/maps`.
pub(super) fn find_cuda_pids(parent_pid: u32, cuda_checkpoint_path: &str) -> Vec<u32> {
    // Include the parent itself — CRIU --tree dumps the root process plus
    // all descendants, and the parent vLLM process often holds a CUDA
    // context (it initialises CUDA before forking the EngineCore workers).
    let mut all_pids = vec![parent_pid];
    all_pids.extend(find_all_descendants(parent_pid));

    // Primary: use cuda-checkpoint --get-state to probe each process.
    // A process with state "running" has an active CUDA context.
    let is_root = unsafe { libc::geteuid() } == 0;
    let cuda_pids: Vec<u32> = all_pids
        .iter()
        .copied()
        .filter(|&pid| {
            let mut cmd = if is_root {
                std::process::Command::new(cuda_checkpoint_path)
            } else {
                let mut c = std::process::Command::new("sudo");
                c.arg(cuda_checkpoint_path);
                c
            };
            let output = cmd
                .args(["--get-state", "--pid", &pid.to_string()])
                .output();
            match output {
                Ok(out) if out.status.success() => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let state = stdout.trim();
                    if state == "running" {
                        debug!(pid, state, "cuda-checkpoint: CUDA context found");
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            }
        })
        .collect();

    if !cuda_pids.is_empty() {
        return cuda_pids;
    }

    // Fallback: check /dev/nvidia device mappings in /proc/PID/maps
    let gpu_pids: Vec<u32> = all_pids
        .iter()
        .copied()
        .filter(|&pid| has_nvidia_mappings(pid))
        .collect();

    if !gpu_pids.is_empty() {
        return gpu_pids;
    }

    // Last resort: return all direct children
    find_child_pids(parent_pid)
}

/// Check if a process has NVIDIA GPU device mappings.
pub(super) fn has_nvidia_mappings(pid: u32) -> bool {
    let maps_path = format!("/proc/{}/maps", pid);
    std::fs::read_to_string(&maps_path)
        .map(|maps| maps.contains("/dev/nvidia"))
        .unwrap_or(false)
}

/// Get direct child PIDs of a process.
fn find_child_pids(pid: u32) -> Vec<u32> {
    let output = std::process::Command::new("pgrep")
        .args(["-P", &pid.to_string()])
        .output()
        .ok();

    match output {
        Some(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .collect(),
        _ => vec![],
    }
}

/// Recursively find all descendant PIDs of a process.
fn find_all_descendants(pid: u32) -> Vec<u32> {
    let mut result = Vec::new();
    let children = find_child_pids(pid);
    for &child in &children {
        result.push(child);
        result.extend(find_all_descendants(child));
    }
    result
}

/// Find all PIDs on the system that have NVIDIA GPU device mappings,
/// excluding any that belong to the given set of known PIDs.
///
/// Uses /proc to scan for processes with /dev/nvidia in their memory maps.
/// This is Linux-only.
#[cfg(unix)]
fn find_stray_gpu_pids(known_pids: &[u32]) -> Vec<u32> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return vec![];
    };

    let mut strays = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if known_pids.contains(&pid) {
            continue;
        }
        if has_nvidia_mappings(pid) {
            strays.push(pid);
        }
    }
    strays
}

impl Orchestrator {
    /// Toggle CUDA suspend/resume via `cuda-checkpoint --toggle`.
    ///
    /// When `suspend` is true, CUDA state is suspended (VRAM copied to host RAM,
    /// GPU memory freed). When false, CUDA state is resumed (copied back to GPU).
    /// The process stays alive throughout.
    ///
    /// With TP>1, toggles all GPU-holding processes in parallel.
    pub(super) async fn cuda_suspend_toggle(
        &self,
        model: &str,
        process: &Arc<Mutex<ManagedProcess>>,
        suspend: bool,
    ) -> Result<(), OrchestratorError> {
        let ckpt_cfg =
            self.checkpoint_config
                .as_ref()
                .ok_or_else(|| OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "CudaSuspend requires checkpoint config (for cuda_checkpoint_path)"
                        .to_string(),
                })?;

        let pids = {
            let guard = process.lock().await;
            if guard.engine_core_pids.is_empty() {
                return Err(OrchestratorError::SleepFailed {
                    model: model.to_string(),
                    reason: "No GPU PIDs available for cuda-checkpoint".to_string(),
                });
            }
            guard.engine_core_pids.clone()
        };

        let action = if suspend { "suspend" } else { "resume" };
        info!(model = %model, pids = ?pids, action, "cuda-checkpoint --toggle ({} process(es), parallel)", pids.len());

        let mut set = tokio::task::JoinSet::new();
        for pid in &pids {
            let pid = *pid;
            let cuda_checkpoint_path = ckpt_cfg.cuda_checkpoint_path.clone();
            set.spawn(async move {
                let output = maybe_sudo(&cuda_checkpoint_path)
                    .args(["--toggle", "--pid", &pid.to_string()])
                    .output()
                    .await
                    .map_err(|e| (pid, format!("Failed to run cuda-checkpoint: {}", e)))?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err((pid, format!("cuda-checkpoint failed: {}", stderr)));
                }
                Ok(pid)
            });
        }

        while let Some(result) = set.join_next().await {
            match result {
                Ok(Ok(pid)) => {
                    debug!(model = %model, pid, action, "cuda-checkpoint toggled");
                }
                Ok(Err((pid, reason))) => {
                    return Err(OrchestratorError::SleepFailed {
                        model: model.to_string(),
                        reason: format!(
                            "cuda-checkpoint --toggle ({}) failed for PID {}: {}",
                            action, pid, reason
                        ),
                    });
                }
                Err(e) => {
                    return Err(OrchestratorError::SleepFailed {
                        model: model.to_string(),
                        reason: format!("cuda-checkpoint task panicked: {}", e),
                    });
                }
            }
        }

        info!(model = %model, action, "cuda-checkpoint toggle succeeded for all {} process(es)", pids.len());
        Ok(())
    }

    /// Kill any GPU-holding processes that don't belong to any managed model.
    ///
    /// This catches zombie processes left behind by OOM kills, failed starts,
    /// or CRIU restore failures that still hold GPU memory and would cause the
    /// next model load to OOM.
    #[cfg(unix)]
    pub(super) async fn kill_stray_gpu_processes(&self) {
        // Collect PIDs of all processes we know about
        let mut known_pids: Vec<u32> = Vec::new();
        for entry in self.processes.iter() {
            let guard = entry.value().lock().await;
            if let Some(ref child) = guard.child
                && let Some(pid) = child.id()
            {
                // Include all descendants of our managed processes
                known_pids.push(pid);
                known_pids.extend(find_all_descendants(pid));
            }
            if let Some(pid) = guard.parent_pid {
                known_pids.push(pid);
                known_pids.extend(find_all_descendants(pid));
            }
            known_pids.extend(guard.engine_core_pids.iter());
        }

        let strays = find_stray_gpu_pids(&known_pids);
        if strays.is_empty() {
            return;
        }

        warn!(
            pids = ?strays,
            "Found {} stray GPU process(es), killing them to free GPU memory",
            strays.len()
        );

        for pid in strays {
            // SAFETY: Killing a process we identified as a stray GPU holder.
            unsafe {
                libc::kill(pid as libc::pid_t, libc::SIGKILL);
            }
            info!(pid, "Killed stray GPU process");
        }

        // Give the kernel a moment to reclaim GPU memory
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
