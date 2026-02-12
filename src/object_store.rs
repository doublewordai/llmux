//! S3-compatible object store for checkpoint images.
//!
//! Uploads checkpoint images after CRIU dump and downloads them
//! before CRIU restore when a local copy is not available.

use crate::config::ObjectStoreConfig;
use anyhow::{Context, Result};
use s3::creds::Credentials;
use s3::{Bucket, Region};
use std::path::Path;
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info};

/// S3 client wrapper for checkpoint operations.
pub struct CheckpointStore {
    bucket: Box<Bucket>,
}

pub struct UploadStats {
    pub files: usize,
    pub bytes: u64,
    pub duration_secs: f64,
}

pub struct DownloadStats {
    pub files: usize,
    pub bytes: u64,
    pub duration_secs: f64,
}

impl CheckpointStore {
    /// Create a new CheckpointStore from config.
    pub fn new(config: &ObjectStoreConfig) -> Result<Self> {
        let region = Region::Custom {
            region: config.region.clone(),
            endpoint: config.endpoint.clone(),
        };
        let credentials = Credentials::new(
            Some(&config.access_key),
            Some(&config.secret_key),
            None,
            None,
            None,
        )
        .context("Failed to create S3 credentials")?;

        let bucket =
            Bucket::new(&config.bucket, region, credentials).context("Failed to create S3 bucket client")?;
        let bucket = bucket.with_path_style();

        Ok(Self { bucket })
    }

    /// Upload all checkpoint images from a local directory to S3.
    ///
    /// S3 key format: `<model>/images/<filename>`
    /// Large files (>1MB) use concurrency 4, small files use concurrency 32.
    pub async fn upload_checkpoint(&self, model: &str, images_dir: &Path) -> Result<UploadStats> {
        let start = std::time::Instant::now();

        // Delete stale files from previous checkpoints.
        // CRIU metadata files have PID-specific names that change between checkpoints,
        // so re-uploading without cleanup accumulates stale files.
        let prefix = format!("{}/images/", model);
        let list = self
            .bucket
            .list(prefix.clone(), None)
            .await
            .context("Failed to list existing S3 objects for cleanup")?;
        let existing: Vec<String> = list
            .iter()
            .flat_map(|r| r.contents.iter())
            .map(|o| o.key.clone())
            .collect();
        if !existing.is_empty() {
            info!(model, files = existing.len(), "Cleaning up previous checkpoint in S3");
            for key in &existing {
                self.bucket
                    .delete_object(key)
                    .await
                    .with_context(|| format!("Failed to delete S3 object: {}", key))?;
            }
        }

        let mut entries = fs::read_dir(images_dir)
            .await
            .with_context(|| format!("Failed to read checkpoint dir: {}", images_dir.display()))?;

        let mut large_files = Vec::new();
        let mut small_files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            if !metadata.is_file() {
                continue;
            }
            let size = metadata.len();
            let path = entry.path();
            if size > 1_048_576 {
                large_files.push((path, size));
            } else {
                small_files.push((path, size));
            }
        }

        let total_files = large_files.len() + small_files.len();
        let total_bytes: u64 = large_files
            .iter()
            .chain(&small_files)
            .map(|(_, size)| size)
            .sum();

        info!(
            model,
            files = total_files,
            size_mb = total_bytes / 1_048_576,
            "Uploading checkpoint to S3"
        );

        // Upload large files with limited concurrency to avoid memory pressure
        let large_sem = Arc::new(tokio::sync::Semaphore::new(4));
        let mut set = tokio::task::JoinSet::new();

        for (path, size) in large_files {
            let key = format!(
                "{}/images/{}",
                model,
                path.file_name().unwrap().to_string_lossy()
            );
            let bucket = self.bucket.clone();
            let sem = large_sem.clone();
            set.spawn(async move {
                let _permit = sem.acquire().await?;
                let data = fs::read(&path).await?;
                bucket.put_object(&key, &data).await?;
                debug!(key, size, "Uploaded large file");
                Ok::<_, anyhow::Error>(())
            });
        }

        // Upload small files with higher concurrency
        let small_sem = Arc::new(tokio::sync::Semaphore::new(32));
        for (path, size) in small_files {
            let key = format!(
                "{}/images/{}",
                model,
                path.file_name().unwrap().to_string_lossy()
            );
            let bucket = self.bucket.clone();
            let sem = small_sem.clone();
            set.spawn(async move {
                let _permit = sem.acquire().await?;
                let data = fs::read(&path).await?;
                bucket.put_object(&key, &data).await?;
                debug!(key, size, "Uploaded small file");
                Ok::<_, anyhow::Error>(())
            });
        }

        while let Some(result) = set.join_next().await {
            result??;
        }

        let duration_secs = start.elapsed().as_secs_f64();
        info!(
            model,
            files = total_files,
            size_mb = total_bytes / 1_048_576,
            duration_secs = format!("{:.1}", duration_secs),
            "Checkpoint uploaded to S3"
        );

        Ok(UploadStats {
            files: total_files,
            bytes: total_bytes,
            duration_secs,
        })
    }

    /// Download checkpoint images from S3 to a local directory.
    ///
    /// Creates the local directory if it doesn't exist.
    pub async fn download_checkpoint(
        &self,
        model: &str,
        images_dir: &Path,
    ) -> Result<DownloadStats> {
        let start = std::time::Instant::now();
        fs::create_dir_all(images_dir)
            .await
            .with_context(|| format!("Failed to create images dir: {}", images_dir.display()))?;

        let prefix = format!("{}/images/", model);
        let list = self
            .bucket
            .list(prefix.clone(), None)
            .await
            .context("Failed to list S3 objects")?;

        let objects: Vec<_> = list.iter().flat_map(|r| r.contents.iter()).collect();

        if objects.is_empty() {
            anyhow::bail!("No checkpoint found in S3 for model '{}'", model);
        }

        let total_files = objects.len();
        let total_bytes: u64 = objects.iter().map(|o| o.size).sum();

        info!(
            model,
            files = total_files,
            size_mb = total_bytes / 1_048_576,
            "Downloading checkpoint from S3"
        );

        let semaphore = Arc::new(tokio::sync::Semaphore::new(8));
        let mut set = tokio::task::JoinSet::new();

        for obj in objects {
            let key = obj.key.clone();
            let filename = key
                .strip_prefix(&prefix)
                .unwrap_or(&key)
                .to_string();
            let dest = images_dir.join(&filename);
            let bucket = self.bucket.clone();
            let sem = semaphore.clone();
            let size = obj.size;

            set.spawn(async move {
                let _permit = sem.acquire().await?;
                let response = bucket.get_object(&key).await?;
                fs::write(&dest, response.as_slice()).await?;
                debug!(key, size, "Downloaded file");
                Ok::<_, anyhow::Error>(())
            });
        }

        while let Some(result) = set.join_next().await {
            result??;
        }

        let duration_secs = start.elapsed().as_secs_f64();
        info!(
            model,
            files = total_files,
            size_mb = total_bytes / 1_048_576,
            duration_secs = format!("{:.1}", duration_secs),
            "Checkpoint downloaded from S3"
        );

        Ok(DownloadStats {
            files: total_files,
            bytes: total_bytes,
            duration_secs,
        })
    }

    /// Check if a checkpoint exists in S3 for the given model.
    pub async fn checkpoint_exists(&self, model: &str) -> Result<bool> {
        let prefix = format!("{}/images/", model);
        let list = self
            .bucket
            .list(prefix, None)
            .await
            .context("Failed to list S3 objects")?;

        Ok(list.iter().any(|r| !r.contents.is_empty()))
    }
}
